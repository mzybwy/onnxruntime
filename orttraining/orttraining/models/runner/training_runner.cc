// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/models/runner/training_runner.h"

#include <algorithm>
#include <memory>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"
#include "orttraining/models/runner/training_util.h"

namespace onnxruntime {
namespace training {

static std::vector<FreeDimensionOverride> overrides = {};
static SessionOptions SESSION_OPTION = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
    false,                             //enable_profiling
    ORT_TSTR(""),                      //optimized_model_filepath
    true,                              //enable_mem_pattern
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level1,          //graph_optimization_level
    {},                                //intra_op_param
    {},                                //inter_op_param
    overrides,                         //free_dimension_overrides
    true,                              //use_per_session_threads
    true                               //thread_pool_allow_spinning
};

TrainingRunner::TrainingRunner(Parameters params, const Environment& env)
    : TrainingRunner(params, env, SESSION_OPTION) {
}

TrainingRunner::TrainingRunner(Parameters params, const Environment& env, SessionOptions session_options)
    : step_(0),
      round_(0),
      weight_update_step_count_(0),
      training_data_set_index_(0),
      params_(params),
      session_options_(session_options),
      session_(session_options, env),
      input_allocator_(params.input_allocator ? params.input_allocator : TrainingUtil::GetCpuAllocator()),
      pipeline_schedule_(params_.num_pipeline_stages),
      pipeline_worker_pool_(params.num_pipeline_stages) {
  ORT_ENFORCE(!params_.model_path.empty());
  if (!params.weights_to_train.empty())
    ORT_ENFORCE(params.weights_not_to_train.empty());
  ORT_ENFORCE(!params_.training_optimizer_name.empty());
  if (params.partition_optimizer)
    ORT_ENFORCE(params.use_nccl, "Optimizer partitioning is only supported with NCCL distributed training.");
  
  if (params_.num_pipeline_stages > 1) {
    pipeline_context_.pipeline_stage_id = params_.mpi_context.world_rank;
    pipeline_context_.num_pipeline_stages = params_.num_pipeline_stages;
    pipeline_context_.num_pipeline_batches = params_.gradient_accumulation_steps - 1;
    pipeline_context_.num_gradient_accumulation_steps = params_.gradient_accumulation_steps;
    pipeline_context_.pipeline_stage_paths = params_.pipeline_stage_paths;
    pipeline_schedule_.add(0, pipeline_context_.num_pipeline_batches);
  }
}

Status TrainingRunner::Initialize() {
  if (params_.num_pipeline_stages > 1 && !pipeline_context_.pipeline_stage_paths.empty()) {
    // Pipeline partition happens outside ORT. We just load the result of partitioning forward graph.
    // Backward graph will be generated using ORT's graph transformers.
    ORT_RETURN_IF_ERROR(session_.Load(pipeline_context_.pipeline_stage_paths[pipeline_context_.pipeline_stage_id]));
  } else {
    ORT_RETURN_IF_ERROR(session_.Load(params_.model_path));
  }

  TrainingSession::TrainingConfiguration config{};
  config.model_with_loss_function_path = params_.model_with_loss_func_path;
  config.model_with_training_graph_path = params_.model_with_training_graph_path;

  config.weight_names_to_train = params_.weights_to_train;
  config.weight_names_to_not_train = params_.weights_not_to_train;
  config.immutable_weights = params_.immutable_weights;

  config.set_gradients_as_graph_outputs = false;

  config.gradient_accumulation_steps = params_.gradient_accumulation_steps;

  config.distributed_config.world_rank = params_.mpi_context.world_rank;
  config.distributed_config.world_size = params_.mpi_context.world_size;
  config.distributed_config.local_size = params_.mpi_context.local_size;
  config.distributed_config.local_rank = params_.mpi_context.local_rank;
  config.distributed_config.data_parallel_size = params_.data_parallel_size;
  config.distributed_config.horizontal_parallel_size = params_.horizontal_parallel_size;
  config.distributed_config.pipeline_stage_size = params_.num_pipeline_stages;

  if (params_.use_mixed_precision) {
    TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_fp16_initializers = params_.use_fp16_initializer;

    config.mixed_precision_config = mp;
  }

  // always configure the loss function
  if (params_.num_pipeline_stages == 1 || params_.mpi_context.world_rank == params_.mpi_context.world_size - 1) {
    TrainingSession::TrainingConfiguration::LossFunctionConfiguration lf{};
    lf.loss_function_info = params_.loss_func_info;

    config.loss_function_config = lf;
  }

  // always configure the optimizer
  {
    TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
    opt.name = params_.training_optimizer_name;
    opt.learning_rate_input_name = params_.lr_params.feed_name;
    opt.weight_attributes_generator = params_.optimizer_attributes;
    opt.weight_int_attributes_generator = params_.optimizer_int_attributes;
    opt.use_fp16_moments = params_.use_fp16_moments;
    opt.do_all_reduce_in_fp16 = params_.allreduce_in_fp16;
    opt.use_nccl = params_.use_nccl;
    opt.partition_optimizer = params_.partition_optimizer;
    opt.adasum_reduction_type = params_.GetAdasumReductionType();
    opt.enable_grad_norm_clip = params_.enable_grad_norm_clip;
    config.optimizer_config = opt;
  }

  if (params_.EnableTensorboard()) {
    TrainingSession::TrainingConfiguration::TensorboardConfiguration tb{};
    tb.summary_name = params_.summary_name;
    tb.scalar_node_names = params_.scalar_names;
    tb.histogram_node_names = params_.histogram_names;
    tb.norm_node_names = params_.norm_names;
    tb.dump_convergence_metrics = params_.dump_convergence_metrics;

    config.tensorboard_config = tb;
  }

  if (params_.use_gist) {
    TrainingSession::TrainingConfiguration::GistConfiguration gist{};

    config.gist_config = gist;
  }

  // Prepare pipeline information to do configuration.
  if (params_.num_pipeline_stages > 1) {
    TrainingSession::TrainingConfiguration::PipelineConfiguration pipe{};
    pipe.num_pipeline_stages = params_.num_pipeline_stages;
    pipe.pipeline_stage_id = params_.mpi_context.world_rank;
    pipe.fetch_names = params_.fetch_names;
    // Do not assign value to config.pipeline_config if pipeline is not used.
    config.pipeline_config = pipe;
  }

  TrainingSession::TrainingConfigurationResult config_result{};

  ORT_RETURN_IF_ERROR(session_.ConfigureForTraining(config, config_result));

  if (config_result.mixed_precision_config_result.has_value()) {
    const std::string& loss_scale_input_name =
        config_result.mixed_precision_config_result.value().loss_scale_input_name;
    if (params_.loss_scale == 0.0f) {
      // use dynamic loss_scale
      loss_scaler_ = onnxruntime::make_unique<LossScaler>(loss_scale_input_name, true, static_cast<float>(1 << 16));
    } else {
      // use static loss_scale
      loss_scaler_ = onnxruntime::make_unique<LossScaler>(loss_scale_input_name, false, params_.loss_scale);
    }
  }

  opt_graph_outputs_ = config_result.opt_config_result.value().output_key_to_graph_output_name;

  // Retrieve pipeline information from configuration result.
  VectorString fetch_names;
  if (params_.num_pipeline_stages > 1) {
    fetch_names = config_result.pipeline_config_result.value().fetch_names;
    // Exposes forward waited event tensor ID name to TrainingRunner.
    // It's an input of a graph.
    pipeline_context_.forward_waited_event_name = config_result.pipeline_config_result.value().forward_waited_event_name;
    // Exposes forward recorded event tensor ID name to TrainingRunner.
    // It's an input of a graph.
    pipeline_context_.forward_recorded_event_name = config_result.pipeline_config_result.value().forward_recorded_event_name;
    // Exposes backward waited event tensor ID name to TrainingRunner.
    // It's an input of a graph.
    pipeline_context_.backward_waited_event_name = config_result.pipeline_config_result.value().backward_waited_event_name;
    // Exposes backward recorded event tensor ID name to TrainingRunner.
    // It's an input of a graph.
    pipeline_context_.backward_recorded_event_name = config_result.pipeline_config_result.value().backward_recorded_event_name;

    pipeline_context_.forward_waited_output_name = config_result.pipeline_config_result.value().forward_waited_output_name;
    pipeline_context_.forward_recorded_output_name = config_result.pipeline_config_result.value().forward_recorded_output_name;
    pipeline_context_.backward_waited_output_name = config_result.pipeline_config_result.value().backward_waited_output_name;
    pipeline_context_.backward_recorded_output_name = config_result.pipeline_config_result.value().backward_recorded_output_name;

    if (!pipeline_context_.forward_waited_output_name.empty()) {
      fetch_names.push_back(pipeline_context_.forward_waited_output_name);
    }

    if (!pipeline_context_.forward_recorded_output_name.empty()) {
      fetch_names.push_back(pipeline_context_.forward_recorded_output_name);
    }

    if (!pipeline_context_.backward_waited_output_name.empty()) {
      fetch_names.push_back(pipeline_context_.backward_waited_output_name);
    }

    if (!pipeline_context_.backward_recorded_output_name.empty()) {
      fetch_names.push_back(pipeline_context_.backward_recorded_output_name);
    }

    // Names of allowed inputs after pipeline partition.
    pipeline_context_.feed_names = config_result.pipeline_config_result.value().feed_names;
    // Names of allowed outputs after pipeline partition.
    pipeline_context_.fetch_names = config_result.pipeline_config_result.value().fetch_names;
  } else {
    fetch_names = params_.fetch_names;
  }

  // Expose all optimizer outputs as graph outputs.
  for (const auto& it : opt_graph_outputs_) {
    fetch_names.push_back(it.second);
  }

  // Expose all optimizer outputs and pipeline outputs and as graph outputs.
  ORT_RETURN_IF_ERROR(session_.OverrideGraphOutputs(fetch_names));

  for (const auto& factory : params_.providers) {
    auto provider = factory.second->CreateProvider();
    ORT_ENFORCE(factory.first == provider->Type());
    ORT_RETURN_IF_ERROR(session_.RegisterExecutionProvider(std::move(provider)));
  }

  if (params_.use_profiler && !session_options_.enable_profiling) {
    // Profiling has not already been enabled, so override from command line options.
    session_.StartProfiling(session_options_.profile_file_prefix);
  }

  ORT_RETURN_IF_ERROR(session_.Initialize());

  // Checkpointing initialization
  // session_.Initialize() must be called prior to LoadCheckpoint()
  if (!params_.checkpoints_dir.empty()) {
    checkpoint_registry_ = onnxruntime::make_unique<CheckpointRegistry>(
        params_.checkpoints_dir, params_.max_num_checkpoints);

    // Load checkpoint, if any
    PathString checkpoint_to_load_path = params_.checkpoint_to_load_path;
    if (!checkpoint_to_load_path.empty() ||
        checkpoint_registry_->TryGetLatestCheckpoint(checkpoint_to_load_path)) {
      ORT_RETURN_IF_ERROR(LoadCheckpoint(checkpoint_to_load_path));
    }
  }

  return Status::OK();
}

Status TrainingRunner::Run(IDataLoader* training_data_loader, IDataLoader* test_data_loader) {
  if (params_.mpi_context.world_rank == 0 && !params_.model_actual_running_graph_path.empty()) {
    session_.Save(params_.model_actual_running_graph_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  // maybe in the future we can support an evaluation-only run
  if (!training_data_loader) {
    LOGS_DEFAULT(WARNING) << "training data loader not provided, nothing to do";
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(TrainingLoop(*training_data_loader, test_data_loader));

  // after successful Run(), update counters
  round_++;
  step_ = 0;

  return Status::OK();
}

// Prepare feeds for a call to one session run.
Status TrainingRunner::PrepareFeedNamesAndFeeds(IDataLoader& training_data_loader,
                                                LearningRateScheduler& lr_scheduler,
                                                const size_t batch_index,
                                                std::vector<std::string>& feed_names,
                                                std::vector<MLValue>& feeds) {
  // Initialize outputs of this function.
  feed_names = std::vector<std::string>();
  feeds = std::vector<MLValue>();

  auto allowed_feed_begin = pipeline_context_.feed_names.begin();
  auto allowed_feed_end = pipeline_context_.feed_names.end();

  // Pick up feeds from data loader
  {
    auto training_data = training_data_loader.CurrentDataSet();
    std::vector<std::string> data_feed_names = training_data_loader.DataSetTensorNames();
    std::vector<MLValue> data_feeds = training_data->GetKthBatch(params_.batch_size, batch_index, input_allocator_);
    for (size_t i = 0; i < data_feed_names.size(); ++i) {
      const auto name = data_feed_names[i];
      if (params_.num_pipeline_stages == 1 || std::find(allowed_feed_begin, allowed_feed_end, name) != allowed_feed_end) {
        feed_names.push_back(name);
        feeds.push_back(data_feeds[i]);
      }
    }
  }

  // Pick up feed from loss scaling.
  if (loss_scaler_) {
    const auto name = loss_scaler_->GetLossScaleInputName();
    if (params_.num_pipeline_stages == 1 || std::find(allowed_feed_begin, allowed_feed_end, name) != allowed_feed_end) {
      feed_names.push_back(name);
      const float loss_scale = loss_scaler_->GetLossScale();
      OrtValue loss_scale_val;
      TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{loss_scale}, &loss_scale_val, input_allocator_);
      feeds.push_back(loss_scale_val);
    }
  }

  // Pick up feed from learning rate schedule.
  {
    const auto name = params_.lr_params.feed_name;
    if (params_.num_pipeline_stages == 1 || std::find(allowed_feed_begin, allowed_feed_end, name) != allowed_feed_end) {
      feed_names.push_back(name);
      const float learning_rate = lr_scheduler.GetLearningRate(step_ + 1);
      OrtValue lr_val;
      TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{learning_rate}, &lr_val, input_allocator_);
      feeds.push_back(lr_val);
    }
  }

  // Create feed of waited event in forward pass.
  if (!pipeline_context_.forward_waited_event_name.empty()) {
    ORT_ENFORCE(params_.num_pipeline_stages > 1);
    feed_names.push_back(pipeline_context_.forward_waited_event_name);
    OrtValue event_id;
    const int64_t id = pipeline_schedule_.get_forward_waited_event_id(
      pipeline_context_.pipeline_stage_id, step_ % pipeline_context_.num_gradient_accumulation_steps);
    TrainingUtil::CreateCpuMLScalar(
      id,
      &event_id,
      input_allocator_);
    feeds.push_back(event_id);
  }

  // Create feed of recorded event in forward pass.
  if (!pipeline_context_.forward_recorded_event_name.empty()) {
    ORT_ENFORCE(params_.num_pipeline_stages > 1);
    feed_names.push_back(pipeline_context_.forward_recorded_event_name);
    OrtValue event_id;
    const int64_t id = pipeline_schedule_.get_forward_recorded_event_id(
      pipeline_context_.pipeline_stage_id, step_ % pipeline_context_.num_gradient_accumulation_steps);
    TrainingUtil::CreateCpuMLScalar(
      id,
      &event_id,
      input_allocator_);
    feeds.push_back(event_id);
  }

  // Create feed of waited event in backward pass.
  if (!pipeline_context_.backward_waited_event_name.empty()) {
    ORT_ENFORCE(params_.num_pipeline_stages > 1);
    feed_names.push_back(pipeline_context_.backward_waited_event_name);
    OrtValue event_id;
    const int64_t id = pipeline_schedule_.get_backward_waited_event_id(
      pipeline_context_.pipeline_stage_id, step_ % pipeline_context_.num_gradient_accumulation_steps);
    TrainingUtil::CreateCpuMLScalar(
      id,
      &event_id,
      input_allocator_);
    feeds.push_back(event_id);
  }

  // Create feed of recorded event in backward pass.
  if (!pipeline_context_.backward_recorded_event_name.empty()) {
    ORT_ENFORCE(params_.num_pipeline_stages > 1);
    feed_names.push_back(pipeline_context_.backward_recorded_event_name);
    OrtValue event_id;
    int64_t id = pipeline_schedule_.get_backward_recorded_event_id(
      pipeline_context_.pipeline_stage_id, step_ % pipeline_context_.num_gradient_accumulation_steps);
    TrainingUtil::CreateCpuMLScalar(
      id,
      &event_id,
      input_allocator_);
    feeds.push_back(event_id);
  }

  return Status::OK();
}

Status TrainingRunner::PrepareFetchNamesAndFetches(const bool do_weight_update,
                                                   std::vector<std::string>& fetch_names,
                                                   std::vector<MLValue>& fetches) {
  // Initialize outputs of this function.
  fetch_names = std::vector<std::string>();
  fetches = std::vector<MLValue>();

  const auto& allowed_fetch_names = pipeline_context_.fetch_names;

  if (do_weight_update) {
    // Set up tensor to be fetched when doing model update. 

    if (params_.num_pipeline_stages > 1) {
      // If pipeline is used, we need to filter out fetches which are not in this pipeline stage.

      for (size_t i = 0; i < params_.fetch_names.size(); ++i) {
        const auto name = params_.fetch_names[i];
        auto it = std::find(allowed_fetch_names.begin(), allowed_fetch_names.end(), name);
        if (it == allowed_fetch_names.end()) {
          continue;
        }
        fetch_names.push_back(name);
      }
    } else {
      // No pipeline. All fetched names should appear in the graph handled by this process.
      fetch_names = params_.fetch_names;
    }

    if (params_.use_mixed_precision) {
      auto it = opt_graph_outputs_.find(OptimizerOutputKey::GradientAllIsFinite);
      ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient norm's IsFinite output is missing in the optimizer output");
      fetch_names.push_back(it->second);
      if (params_.use_adasum) {
        it = opt_graph_outputs_.find(OptimizerOutputKey::DeltaAllIsFinite);
        ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Adasum delta's IsFinite output is missing in the optimizer output");
        fetch_names.push_back(it->second);
      }
    }
  } else {
    // Set up tensor to be fetched when doing gradient accumulation. 

    if (params_.gradient_accumulation_steps > 1) {
      auto it = opt_graph_outputs_.find(OptimizerOutputKey::GradientAccumulation);
      ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient accumulation output is missing in the optimizer output");
      fetch_names.push_back(it->second);
    }

    // Always execute event operators to avoid deadlock if pipeline is used.
    // TODO: create a list of must-to-fetch tensors and pass it to all graph transformer.
    if (params_.num_pipeline_stages) {
        if (!pipeline_context_.forward_waited_output_name.empty()) {
          fetch_names.push_back(pipeline_context_.forward_waited_output_name);
        }
        if (!pipeline_context_.forward_recorded_output_name.empty()) {
          fetch_names.push_back(pipeline_context_.forward_recorded_output_name);
        }
        if (!pipeline_context_.backward_waited_output_name.empty()) {
          fetch_names.push_back(pipeline_context_.backward_waited_output_name);
        }
        if (!pipeline_context_.backward_recorded_output_name.empty()) {
          fetch_names.push_back(pipeline_context_.backward_recorded_output_name);
        }
    }
  }

  // We need to fetch at least one variable.
  // If there is nothing to fetch, we fetch all model outputs.
  if (fetch_names.empty()) {
    fetch_names = allowed_fetch_names;
  }

  return Status::OK();
}

Status TrainingRunner::RunWithUpdate(VectorString& feed_names,
                                     VectorString& fetch_names,
                                     std::vector<MLValue>& feeds,
                                     std::vector<MLValue>& fetches) {
  pipeline_worker_pool_.join_all();

  ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                    feed_names,
                                    feeds,
                                    fetch_names,
                                    &fetches));

  if (loss_scaler_) {
    auto it = std::find(fetch_names.begin(), fetch_names.end(), opt_graph_outputs_[OptimizerOutputKey::GradientAllIsFinite]);
    if (it != fetch_names.end()) {
      const size_t index = static_cast<size_t>(std::distance(fetch_names.begin(), it));
      const Tensor& all_is_finite_t = fetches[index].Get<Tensor>();
      const bool is_all_finite = *(all_is_finite_t.template Data<bool>());
      loss_scaler_->UpdateLossScale(is_all_finite);
    }
  }

  // Assume that only the last pipeline stage can see loss, predicted value, and so on.
  // Thus, the error function should only be called when we are at the last stage.
  if (params_.num_pipeline_stages == 1 ||
      pipeline_context_.pipeline_stage_id == pipeline_context_.num_pipeline_stages - 1 &&
      !params_.is_perf_test &&
      weight_update_step_count_ % params_.display_loss_steps == 0) {
    if (params_.error_function) {
      params_.error_function(feed_names, feeds, fetch_names, fetches, weight_update_step_count_);
    }
    if (params_.post_evaluation_callback) {
      params_.post_evaluation_callback(params_.batch_size, weight_update_step_count_, "train");
    }
  }

  // Add one after process one batch.
  ++step_;
  // Add one after update the model once.
  ++weight_update_step_count_;

  return Status::OK();
}

Status TrainingRunner::RunWithoutUpdate(VectorString& feed_names,
                                        VectorString& fetch_names,
                                        std::vector<MLValue>& feeds,
                                        size_t& gradient_accumulation_step_count) {
  const size_t worker_id = step_ % pipeline_context_.num_pipeline_stages;
  pipeline_worker_pool_.join(worker_id);
  pipeline_worker_pool_.worker_states[worker_id].feeds = feeds;
  pipeline_worker_pool_.worker_states[worker_id].feed_names = feed_names;
  pipeline_worker_pool_.worker_states[worker_id].fetch_names = fetch_names;
  pipeline_worker_pool_.worker_states[worker_id].fetches = std::vector<MLValue>();

  pipeline_worker_pool_.workers[worker_id] = std::thread([&](
    const size_t worker_id) {
    RunOptions run_options;
    run_options.only_execute_path_to_fetches = true;
    session_.Run(
      run_options,
      pipeline_worker_pool_.worker_states[worker_id].feed_names,
      pipeline_worker_pool_.worker_states[worker_id].feeds,
      pipeline_worker_pool_.worker_states[worker_id].fetch_names,
      &(pipeline_worker_pool_.worker_states[worker_id].fetches));
  }, worker_id);

  // Add one after process one batch.
  ++step_;
  // Add one after comuting one forward-backward path without applying optimizer.
  ++gradient_accumulation_step_count;

  return Status::OK();
}

Status TrainingRunner::TrainingLoop(IDataLoader& training_data_loader, IDataLoader* test_data_loader) {
  const bool enable_checkpoint_saving =
      params_.mpi_context.world_rank == 0 &&
      checkpoint_registry_ && params_.checkpoint_period > 0;

  if (test_data_loader) {
    ORT_RETURN_IF_ERROR(test_data_loader->InitializeDataSetIndex(0));
  }
  ORT_RETURN_IF_ERROR(training_data_loader.InitializeDataSetIndex(training_data_set_index_));

  const size_t num_shards_to_visit = training_data_loader.NumShards();
  const auto lr_scheduler = LearningRateScheduler::Create(params_.lr_params, params_.num_train_steps);

  double total_time{0};
  size_t epoch = 0;  // Note: epoch is not set properly when loaded from a checkpoint, but it's only for display.
  size_t gradient_accumulation_step_count = 0;
  const auto step_start = step_;
  const auto weight_update_step_count_start = weight_update_step_count_;

  // how many steps at last we used for stabilized perf benchmarking.
  const size_t stabilized_perf_total_step_count = std::min(static_cast<size_t>(128), params_.num_train_steps);
  const size_t stabilized_perf_start_step = params_.num_train_steps - stabilized_perf_total_step_count;
  double stabilized_total_time{0};

  while (step_ < params_.num_train_steps) {
    for (size_t shard_it = 0; shard_it < num_shards_to_visit; ++shard_it) {
      auto training_data = training_data_loader.CurrentDataSet();
      training_data_set_index_ = training_data_loader.CurrentDataSetIndex();
      if (training_data == nullptr) {
        printf("Skipping shard at index %d, which failed to load.\n",
               static_cast<int>(training_data_loader.CurrentDataSetIndex()));
        training_data_loader.MoveToNextDataSet();
        continue;
      }

      // Shuffle the data for each epoch
      if (params_.shuffle_data) {
        printf("Randomly shuffle training data.\n");
        training_data->RandomShuffle();
      }

      // loop through the data
      size_t batch_num_cur_shard = training_data->TotalBatch(params_.batch_size);
      for (size_t batch = 0; batch < batch_num_cur_shard && step_ < params_.num_train_steps; ++batch) {
        const bool is_weight_update_step = (step_ + 1) % params_.gradient_accumulation_steps == 0;

        VectorString feed_names;
        VectorString fetch_names;
        std::vector<MLValue> feeds;
        std::vector<MLValue> fetches;

        PrepareFeedNamesAndFeeds(training_data_loader, *lr_scheduler, batch, feed_names, feeds);

        PrepareFetchNamesAndFetches(is_weight_update_step, fetch_names, fetches);

        auto start = std::chrono::high_resolution_clock::now();

        if (is_weight_update_step) {
          RunWithUpdate(feed_names, fetch_names, feeds, fetches);
        } else {
          RunWithoutUpdate(feed_names, fetch_names, feeds, gradient_accumulation_step_count); 
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seconds = end - start;
        total_time += duration_seconds.count();
        if (step_ >= stabilized_perf_start_step) {
          stabilized_total_time += duration_seconds.count();
        }

        // Print some info when reaching the end of the batch.
        // When pipeline is enabled, the first stage, idexed by 0, computes the start and end of a batch.
        if (params_.num_pipeline_stages == 1 || pipeline_context_.pipeline_stage_id == 0) {
          printf("Round %d, Step: %d, epoch: %d, batch: %d/%d, shard_iteration: %d/%d, time: %.2f ms, throughput: %.2f ex/sec \n",
                static_cast<int>(round_),
                static_cast<int>(step_),
                static_cast<int>(epoch),
                static_cast<int>(batch),
                static_cast<int>(batch_num_cur_shard),
                static_cast<int>(shard_it + 1),
                static_cast<int>(num_shards_to_visit),
                duration_seconds.count() * 1000,
                params_.batch_size * (step_ - step_start) / total_time);
          printf("Training data range: [%d - %d)\n",
                static_cast<int>(batch * params_.batch_size),
                static_cast<int>((batch + 1) * params_.batch_size - 1));
        }

        if (test_data_loader &&
            params_.do_eval && step_ % params_.evaluation_period == 0) {
          ORT_RETURN_IF_ERROR(Evaluate(session_, *test_data_loader));
        }

        if (enable_checkpoint_saving && is_weight_update_step &&
            weight_update_step_count_ % params_.checkpoint_period == 0) {
          PathString new_checkpoint_path, old_checkpoint_path;
          bool should_remove_old_checkpoint;

          ORT_RETURN_IF_ERROR(checkpoint_registry_->AddCheckpoint(
              weight_update_step_count_, new_checkpoint_path,
              should_remove_old_checkpoint, old_checkpoint_path));

          // ensure checkpoint directory exists
          if (!Env::Default().FolderExists(params_.checkpoints_dir)) {
            ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(params_.checkpoints_dir));
          }

          if (should_remove_old_checkpoint) {
            const auto status = Env::Default().DeleteFolder(old_checkpoint_path);
            LOGS_DEFAULT_IF(!status.IsOK(), WARNING)
                << "Failed to delete old checkpoint. "
                << "Path: " << ToMBString(old_checkpoint_path)
                << ", error: " << status.ErrorMessage();
          }

          ORT_RETURN_IF_ERROR(SaveCheckpoint(new_checkpoint_path));
        }
      }  // end of one file/shard

      pipeline_worker_pool_.join_all();
      if (step_ < params_.num_train_steps) {
        training_data_loader.MoveToNextDataSet();
      }
    }  // end of one epoch

    epoch++;
  }

  // Print some info when reaching the end of the batch.
  // For pipeline, the first stage, idexed by 0, computes the start and end of each batch.
  // That is, it computes the start and the end even for the last, and we threfore should
  // print out timing result at the first stage.
  if (pipeline_context_.pipeline_stage_id == 0) {
    std::cout << "Round: " << round_ << "\n"
              << "Batch size: " << params_.batch_size << "\n"
              << "Number of Batches: " << (step_ - step_start) << "\n"
              << "Gradient Accumulation Steps: " << gradient_accumulation_step_count << "\n"
              << "Weight Update Steps: " << (weight_update_step_count_ - weight_update_step_count_start) << "\n"
              << "Total Running Time: " << total_time << " Seconds \n"
              << "Average Running Time Per Batch: " << total_time / (step_ - step_start) * 1000 << " ms\n"
              << "Throughput: " << params_.batch_size * (step_ - step_start) / total_time << " Examples / Second\n"
              << "Stabilized Throughput: " << params_.batch_size / (stabilized_total_time / stabilized_perf_total_step_count)
              << " Examples / Second\n";
  }
  return Status::OK();
}

Status TrainingRunner::EndTraining(IDataLoader* data_loader) {
  if (params_.use_profiler) {
    // Write profiler data to disk.
    // We do this first in case there are any problems saving the trained model.
    std::string profile_file = session_.EndProfiling();
    std::cout << "Profiler data written to file " << profile_file << "\n";
  }

  if (params_.mpi_context.world_rank != 0) {
    printf("Skipping end-training on Device #%d, as it's not the root.\n", params_.mpi_context.world_rank);
    return Status::OK();
  }

  if (params_.num_pipeline_stages == 1 && data_loader) {
    // Test the in-memory model before saving.
    printf("\nEvaluating the final model on the test set.\n");
    ORT_RETURN_IF_ERROR(Evaluate(session_, *data_loader));
  }

  if (params_.output_dir.empty()) {
    printf("No output directory specified, skipping save of trained model.\n");
    return Status::OK();
  }

  // Create output directory if needed.
  if (!params_.output_dir.empty()) {
    ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(params_.output_dir));
  }

  printf("\nSaving the trained model.\n");
  const PathString model_base_name = GetLastComponent(params_.model_path);

  const PathString trained_model_path =
      params_.output_dir + GetPathSep<PathChar>() + model_base_name + ORT_TSTR("_trained.onnx");
  ORT_RETURN_IF_ERROR(session_.Save(
      trained_model_path, TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS));

  const PathString trained_model_with_loss_func_path =
      params_.output_dir + GetPathSep<PathChar>() + model_base_name + ORT_TSTR("_with_cost_trained.onnx");
  ORT_RETURN_IF_ERROR(session_.Save(
      trained_model_with_loss_func_path, TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));

  return Status::OK();
}

Status TrainingRunner::Evaluate(InferenceSession& session, IDataLoader& data_loader) {
  if (params_.skip_evaluation) {
    printf("Skipping evaluation...\n");
    return Status::OK();
  }

  if (params_.mpi_context.world_rank != 0) {
    printf("Skipping evaluation on Device #%d, as it's not the root.\n", params_.mpi_context.world_rank);
    return Status::OK();
  }

  // A static batch index representing current test batch
  static size_t current_batch = 0;
  std::vector<std::string> feed_names = data_loader.DataSetTensorNames();
  if (loss_scaler_) {
    feed_names.push_back(loss_scaler_->GetLossScaleInputName());
  }
  feed_names.push_back(params_.lr_params.feed_name);
  auto test_data = data_loader.CurrentDataSet();
  if (params_.shuffle_data && current_batch == 0) {
    printf("Randomly shuffle test data.\n");
    test_data->RandomShuffle();
  }

  const size_t evaluation_batch_size = params_.eval_batch_size;

  printf("Test data range: [%d - %d)\n",
         static_cast<int>(current_batch * evaluation_batch_size),
         static_cast<int>((current_batch + 1) * evaluation_batch_size - 1));

  const size_t num_batches = size_t(ceil((float)evaluation_batch_size / (float)params_.batch_size));
  if (evaluation_batch_size % params_.batch_size != 0) {
    printf(
        "WARNING: evaluation_batch_size %zu is not an integer multiple of batch_size %zu. "
        "Using evaluation_batch_size %zu\n",
        evaluation_batch_size,
        params_.batch_size,
        num_batches * params_.batch_size);
  }

  OrtValue loss_scale_val;
  TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{params_.loss_scale}, &loss_scale_val);

  RunOptions run_options;
  run_options.only_execute_path_to_fetches = true;
  for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<MLValue> feeds = test_data->GetKthBatch(params_.batch_size, current_batch);
    if (loss_scaler_) {
      feeds.push_back(loss_scale_val);
    }
    OrtValue lr_ort_val;
    TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{params_.lr_params.initial_lr}, &lr_ort_val);
    feeds.push_back(lr_ort_val);
    std::vector<MLValue> fetches;
    ORT_RETURN_IF_ERROR(session.Run(run_options,
                                    feed_names,
                                    feeds,
                                    params_.fetch_names,
                                    &fetches));

    // Call error function
    if (params_.error_function) {
      params_.error_function(feed_names, feeds, params_.fetch_names, fetches, step_);
    }

    // Set to next batch
    if (++current_batch >= test_data->TotalBatch(params_.batch_size)) {
      // Move to next shard
      test_data = data_loader.MoveToNextDataSet();
      current_batch = 0;
    }
  }

  // Call after a test batch.
  if (params_.post_evaluation_callback) {
    params_.post_evaluation_callback(evaluation_batch_size, step_, "test");
  }

  return Status::OK();
}

Status TrainingRunner::SaveCheckpoint(const PathString& checkpoint_path) {
  NameMLValMap checkpointed_tensors{};
  ORT_RETURN_IF_ERROR(session_.GetStateTensors(checkpointed_tensors));

  std::unordered_map<std::string, std::string> checkpointed_properties{};
  ORT_RETURN_IF_ERROR(SaveCheckpointProperties(checkpointed_properties));

  ORT_RETURN_IF_ERROR(SaveModelCheckpoint(
      checkpoint_path, session_.GetDataTransferManager(),
      checkpointed_tensors, checkpointed_properties));

  return Status::OK();
}

namespace {
Status WithOrtValuesFromTensorProtos(
    const PathString& model_location,
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    std::function<Status(const NameMLValMap&)> use_name_to_ort_value_fn) {
  static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};

  NameMLValMap name_to_ort_value{};
  std::vector<std::vector<char>> tensor_buffers{};
  std::vector<ScopedOrtCallbackInvoker> tensor_deleters{};

  for (const auto& tensor_proto : tensor_protos) {
    const auto* tensor_type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type());
    const size_t element_size = tensor_type->GetElementType()->Size();
    const TensorShape shape{
        tensor_proto.dims().data(), static_cast<size_t>(tensor_proto.dims().size())};

    std::vector<char> tensor_buffer{};
    tensor_buffer.resize(element_size * shape.Size());

    const MemBuffer mem_buffer{tensor_buffer.data(), tensor_buffer.size(), cpu_alloc_info};

    OrtValue ort_value;
    OrtCallback callback;

    ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(
        Env::Default(), model_location.c_str(), tensor_proto, mem_buffer,
        ort_value, callback));
    ScopedOrtCallbackInvoker callback_invoker{callback};

    name_to_ort_value.emplace(tensor_proto.name(), ort_value);
    tensor_buffers.emplace_back(std::move(tensor_buffer));
    tensor_deleters.emplace_back(std::move(callback_invoker));
  }

  ORT_RETURN_IF_ERROR(use_name_to_ort_value_fn(name_to_ort_value));

  return Status::OK();
}
}  // namespace

Status TrainingRunner::LoadCheckpoint(const PathString& checkpoint_path) {
  std::vector<ONNX_NAMESPACE::TensorProto> checkpointed_tensors{};
  std::unordered_map<std::string, std::string> checkpointed_properties{};
  ORT_RETURN_IF_ERROR(LoadModelCheckpoint(
      checkpoint_path, session_.GetModelLocation(),
      checkpointed_tensors, checkpointed_properties));

  ORT_RETURN_IF_ERROR(WithOrtValuesFromTensorProtos(
      session_.GetModelLocation(), checkpointed_tensors,
      [this](const NameMLValMap& name_to_ort_value) -> Status {
        ORT_RETURN_IF_ERROR(session_.SetStateTensors(name_to_ort_value, true));
        return Status::OK();
      }));

  ORT_RETURN_IF_ERROR(LoadCheckpointProperties(checkpointed_properties));

  return Status::OK();
}

namespace {
namespace property_names {
constexpr const char* k_step = "step";
constexpr const char* k_round = "round";
constexpr const char* k_weight_update_step = "weight_update_step";
constexpr const char* k_training_data_set_index = "training_data_set_index";
constexpr const char* k_loss_scaler_state = "loss_scaler_state";
}  // namespace property_names

template <typename T>
Status FromString(const std::string& s, T& t) {
  std::istringstream i{s};
  ORT_RETURN_IF_NOT(i >> t && i.eof());
  return Status::OK();
}
}  // namespace

Status TrainingRunner::SaveCheckpointProperties(
    std::unordered_map<std::string, std::string>& properties) const {
  auto save_property = [&properties](const char* name, auto val) {
    properties[name] = std::to_string(val);
  };

  save_property(property_names::k_step, step_);
  save_property(property_names::k_round, round_);
  save_property(property_names::k_weight_update_step, weight_update_step_count_);
  save_property(property_names::k_training_data_set_index, training_data_set_index_);

  if (loss_scaler_) {
    properties[property_names::k_loss_scaler_state] = loss_scaler_->SaveToString();
  }

  return Status::OK();
}

Status TrainingRunner::LoadCheckpointProperties(
    const std::unordered_map<std::string, std::string>& properties) {
  auto load_property = [&properties](const char* name, auto& val) {
    auto prop_it = properties.find(name);
    ORT_RETURN_IF_NOT(prop_it != properties.end());
    ORT_RETURN_IF_ERROR(FromString(prop_it->second, val));
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(load_property(property_names::k_step, step_));
  ORT_RETURN_IF_ERROR(load_property(property_names::k_round, round_));
  ORT_RETURN_IF_ERROR(load_property(
      property_names::k_weight_update_step, weight_update_step_count_));
  ORT_RETURN_IF_ERROR(load_property(
      property_names::k_training_data_set_index, training_data_set_index_));

  if (loss_scaler_) {
    auto prop_it = properties.find(property_names::k_loss_scaler_state);
    ORT_RETURN_IF_NOT(prop_it != properties.end());
    ORT_RETURN_IF_ERROR(loss_scaler_->LoadFromString(prop_it->second));
  }

  return Status::OK();
}

Status TrainingRunner::UpdateParams(Parameters params) {
  params_.lr_params.initial_lr = params.lr_params.initial_lr;
  params_.lr_params.warmup_ratio = params.lr_params.warmup_ratio;
  params_.num_train_steps = params.num_train_steps;
  params_.batch_size = params.batch_size;
  params_.gradient_accumulation_steps = params.gradient_accumulation_steps;
  return Status::OK();
}

Status TrainingRunner::ResetLossScaler() {
  if (loss_scaler_) {
    loss_scaler_->Reset();
  }
  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
