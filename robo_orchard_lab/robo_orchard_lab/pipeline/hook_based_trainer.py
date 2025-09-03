# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
基于钩子的训练器模块
功能：提供灵活的、可扩展的训练流程管理
特点：通过钩子机制实现训练过程的精细控制和定制化扩展
"""

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from robo_orchard_core.utils.config import Config
from torch.utils.data import DataLoader

# 导入RoboOrchard核心组件
from robo_orchard_lab.pipeline.batch_processor.mixin import BatchProcessorMixin
from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHookConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)
from robo_orchard_lab.pipeline.hooks.optimizer import OptimizerHookConfig
from robo_orchard_lab.pipeline.hooks.validation import ValidationHookConfig
from robo_orchard_lab.utils.huggingface import (
    AcceleratorState,
    accelerator_load_state,
)

import pdb


# 获取当前模块的日志记录器
logger = get_logger(__name__)

# 导出的公共类列表
__all__ = [
    "HookBasedTrainer",
    "ResumeCheckpointConfig", 
    "GradientClippingHookConfig",
    "ValidationHookConfig",
    "PipelineHookOrConfigType",
]


@dataclass
class TrainerProgressState(AcceleratorState):
    """
    训练进度状态类
    
    继承自AcceleratorState，用于存储和管理训练过程中的状态信息。
    该类设计用于与Trainer类配合使用，跟踪当前的轮次、步骤等信息。
    
    Attributes:
        epoch_id (int): 当前轮次，从0开始
        step_id (int): 当前轮次内的步骤数，从0开始
        global_step_id (int): 跨所有轮次的总步数，从0开始
    """

    epoch_id: int = 0
    """当前轮次。从0开始计数"""
    
    step_id: int = 0
    """当前步骤。每轮从0开始计数"""
    
    global_step_id: int = 0
    """全局步数。跨所有轮次的累计步数，从0开始"""

    def update_step(self) -> None:
        """
        更新步骤计数器
        
        将step_id和global_step_id都增加1。
        每完成一个训练步骤后调用此方法。
        """
        self.step_id += 1
        self.global_step_id += 1

    def update_epoch(self) -> None:
        """
        更新轮次计数器
        
        将epoch_id增加1，并将step_id重置为0。
        每完成一个训练轮次后调用此方法。
        """
        self.epoch_id += 1
        self.step_id = 0

    def is_training_end(
        self, max_step: int | None, max_epoch: int | None
    ) -> bool:
        """
        检查训练循环是否应该结束
        
        根据当前状态判断是否达到了指定的最大步数或轮次。
        如果当前步数或轮次超过了指定的最大值，返回True。
        如果max_step和max_epoch都为None，返回False。
        
        Args:
            max_step (int|None): 允许的最大步数
            max_epoch (int|None): 允许的最大轮次
            
        Returns:
            bool: 如果训练循环应该结束返回True，否则返回False
        """
        # 检查是否达到最大步数
        if max_step is not None and self.global_step_id >= max_step:
            return True
        # 检查是否达到最大轮次
        if max_epoch is not None and self.epoch_id >= max_epoch:
            return True
        return False

    def sync_pipeline_hook_arg(self, hook_args: PipelineHookArgs) -> None:
        """
        同步训练状态到钩子参数
        
        将当前的训练状态（epoch、step、global_step）同步到提供的钩子参数中，
        使钩子函数能够访问到最新的训练状态信息。
        
        Args:
            hook_args (PipelineHookArgs): 要同步的钩子参数对象
        """
        hook_args.epoch_id = self.epoch_id
        hook_args.step_id = self.step_id
        hook_args.global_step_id = self.global_step_id


class ResumeCheckpointConfig(Config):
    """
    恢复检查点的配置类
    
    用于配置从检查点恢复训练的相关参数，支持本地和远程检查点。
    """

    resume_from: str
    """包含检查点的目录路径"""
    
    cache_dir: str | None = None
    """如果从远程路径加载，用于缓存检查点的目录"""

    safe_serialization: bool = True
    """加载状态时是否使用安全序列化
    
    当input_dir是远程路径时使用。检查点文件的名称取决于
    `safe_serialization`是否设置为True或False。用户应确保
    远程目录中的检查点文件与指定的`safe_serialization`选项兼容。
    """

    def load_state(self, accelerator: Accelerator, **kwargs) -> None:
        """
        从检查点加载accelerator的状态
        
        Args:
            accelerator (Accelerator): 要加载状态的`Accelerator`实例
            **kwargs: 传递给加载函数的额外参数
        """
        accelerator_load_state(
            accelerator=accelerator,
            input_dir=self.resume_from,
            cache_dir=self.cache_dir,
            safe_serialization=self.safe_serialization,
            **kwargs,
        )


# 管道钩子或配置类型的联合类型定义
PipelineHookOrConfigType = PipelineHooksConfig | PipelineHooks


class HookBasedTrainer:
    """
    基于钩子的训练器类
    
    使用钩子来管理训练过程的训练器。数据加载器、模型、优化器和学习率调度器
    都通过`Accelerator`实例进行准备，提供分布式训练能力。`PipelineHooks`
    用于管理训练过程，允许为训练循环的各个阶段定义自定义钩子。
    
    完整的训练过程与钩子结构如下：
    
    .. code-block:: text
    
        with on_loop:
            with on_epoch:
                for batch in dataloader:
                    with on_step:
                        with on_batch:
                            batch_processor(...)
                            ...
                        update step id
                update epoch id
    
    注意：
        训练器将按顺序注册以下默认钩子：
        
        - `GradientClippingHook`: 负责裁剪梯度以防止梯度爆炸。
          如果提供了`grad_clip`参数，将注册此钩子。
          
        - `OptimizerHook`: 负责执行优化步骤和更新学习率调度器。
        
        - `ValidationHook`: 负责在训练期间执行验证。
          将以指定频率调用评估回调函数。
          如果提供了`validation`参数，将注册此钩子。
    
    Args:
        accelerator (Accelerator): 管理分布式训练的`Accelerator`实例
        model (torch.nn.Module): 要训练的模型
        dataloader (DataLoader | Iterable): 训练期间向模型提供批次的数据加载器
        batch_processor (BatchProcessorMixin): 负责处理批次和反向传播损失的批处理器
        optimizer (torch.optim.Optimizer): 训练期间使用的优化器
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): 训练期间使用的学习率调度器
        hooks (PipelineHooks | Iterable[PipelineHooks]): 训练期间使用的钩子
        max_step (int | None): 训练的最大步数。必须指定`max_step`或`max_epoch`
        max_epoch (int | None): 训练的最大轮次。必须指定`max_step`或`max_epoch`
        grad_clip (GradClipConfig | None): 梯度裁剪配置
        validation (ValidationConfig | None): 验证配置。如果未指定，不会执行验证
        resume_from (ResumeCheckpointConfig | None): 从检查点恢复的配置
    """

    def __init__(
        self,
        accelerator: Accelerator,                                # 分布式训练管理器
        model: torch.nn.Module,                                  # 训练模型
        dataloader: DataLoader | Iterable,                      # 数据加载器
        batch_processor: BatchProcessorMixin,                   # 批处理器
        optimizer: torch.optim.Optimizer,                       # 优化器
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,     # 学习率调度器
        hooks: PipelineHookOrConfigType | Iterable[PipelineHookOrConfigType], # 钩子配置
        max_step: int | None = None,                            # 最大步数
        max_epoch: int | None = None,                           # 最大轮次
        grad_clip: GradientClippingHookConfig | None = None,    # 梯度裁剪配置
        validation: ValidationHookConfig | None = None,         # 验证配置
        resume_from: ResumeCheckpointConfig | None = None,      # 恢复配置
    ):
        # === 参数验证 ===
        
        # 确保至少指定了max_step或max_epoch中的一个
        if max_step is None and max_epoch is None:
            raise ValueError(
                "Either `max_step` or `max_epoch` must be specified."
            )
        
        # 验证max_step的有效性
        if max_step is not None and max_step < 1:
            raise ValueError(
                "max_step = {} < 1 is not allowed".format(max_step)
            )
        
        # 验证max_epoch的有效性
        if max_epoch is not None and max_epoch < 1:
            raise ValueError(
                "max_epoch = {} < 1 is not allowed".format(max_epoch)
            )

        # === 基础属性初始化 ===
        
        self.accelerator = accelerator  # 保存accelerator实例
        user_hooks = PipelineHooks.from_hooks(hooks)  # 转换用户提供的钩子
        self.max_step = max_step        # 最大步数
        self.max_epoch = max_epoch      # 最大轮次

        # === 使用accelerator准备训练组件 ===
        # accelerator.prepare()会将组件包装为分布式训练兼容的版本
        
        self.model: torch.nn.Module = accelerator.prepare(model)
        """准备后的模型：支持分布式训练"""
        
        self.dataloader: DataLoader = accelerator.prepare(dataloader)
        """准备后的数据加载器：支持分布式数据采样"""
        
        self.optimizer: AcceleratedOptimizer = accelerator.prepare(optimizer)
        """准备后的优化器：支持分布式梯度同步"""
        
        self.lr_scheduler: AcceleratedScheduler = accelerator.prepare(lr_scheduler)
        """准备后的学习率调度器：与分布式训练兼容"""
        
        # 初始化训练进度状态
        self.trainer_progress_state = TrainerProgressState()
        # 注册训练状态用于检查点保存和加载
        accelerator.register_for_checkpointing(self.trainer_progress_state)

        # 保存批处理器（不需要accelerator准备）
        self.batch_processor = batch_processor

        # === 钩子系统初始化 ===
        
        self.hooks = PipelineHooks()  # 创建空的钩子集合
        
        # 注册默认钩子：梯度裁剪、优化器、验证
        # 注意：钩子的注册顺序很重要，它们会按注册顺序执行
        
        if grad_clip is not None:
            # 如果提供了梯度裁剪配置，注册梯度裁剪钩子
            self.hooks += grad_clip()
        
        # 注册优化器钩子（必需的，负责参数更新和学习率调度）
        self.hooks += OptimizerHookConfig()()
        
        if validation is not None:
            # 如果提供了验证配置，注册验证钩子
            self.hooks += validation()

        # 注册用户自定义钩子（在默认钩子之后执行）
        self.hooks += user_hooks

        # === 恢复训练状态初始化 ===
        
        self._start_epoch = 0  # 起始轮次
        self._start_step = 0   # 起始步骤

        # 如果提供了恢复配置，从检查点恢复状态
        if resume_from is not None:
            logger.info(f"Resume from: {resume_from}", main_process_only=True)
            # 加载检查点状态
            resume_from.load_state(accelerator=self.accelerator)
            # 更新起始位置为检查点中的状态
            self._start_epoch = self.trainer_progress_state.epoch_id
            self._start_step = self.trainer_progress_state.step_id

    def _get_hook_args(self, **kwargs) -> PipelineHookArgs:
        """
        获取钩子参数
        
        创建并返回包含当前训练状态和额外参数的HookArgs对象。
        这些参数会传递给所有钩子函数。
        
        Args:
            **kwargs: 要包含在HookArgs对象中的额外参数
            
        Returns:
            PipelineHookArgs: 包含当前训练状态和额外参数的对象
        """
        # 创建钩子参数对象，包含训练器的所有重要状态
        hookargs = PipelineHookArgs(
            accelerator=self.accelerator,                           # 分布式训练管理器
            max_step=self.max_step,                                # 最大步数
            max_epoch=self.max_epoch,                              # 最大轮次
            epoch_id=self.trainer_progress_state.epoch_id,         # 当前轮次
            step_id=self.trainer_progress_state.step_id,           # 当前步骤
            global_step_id=self.trainer_progress_state.global_step_id, # 全局步数
            dataloader=self.dataloader,                            # 数据加载器
            optimizer=self.optimizer,                              # 优化器
            lr_scheduler=self.lr_scheduler,                        # 学习率调度器
            start_epoch=self._start_epoch,                         # 起始轮次
            start_step=self._start_step,                           # 起始步骤
        )
        
        # 添加额外的关键字参数
        for k, v in kwargs.items():
            setattr(hookargs, k, v)
            
        return hookargs

    def __call__(self):
        """
        执行完整的训练流程
        
        这是训练器的主要入口点，实现了基于钩子的训练循环。
        训练流程包括嵌套的钩子上下文：loop -> epoch -> step -> batch。
        """
        
        # === 训练开始日志 ===
        
        logger.info(
            "\n" + "=" * 50 + "BEGIN TRAINING" + "=" * 50,
            main_process_only=True,  # 只在主进程打印
        )
        logger.info(
            f"Start training loop from epoch {self._start_epoch} "
            f"and step {self._start_step}",
            main_process_only=True,
        )
        
        # === 训练准备 ===
        
        end_loop_flag = False  # 训练结束标志
        self.model.train()     # 设置模型为训练模式

        def step(
            batch: Any,
            batch_processor: BatchProcessorMixin,
        ):
            """
            单步训练函数
            
            处理单个批次的训练，包括前向传播、损失计算和反向传播。
            这个函数在step和batch钩子的上下文中执行。
            
            Args:
                batch: 当前批次的数据
                batch_processor: 批处理器，负责实际的训练逻辑
            """
            # 进入step钩子上下文
            with self.hooks.begin(
                "on_step", self._get_hook_args()
            ) as on_step_hook_args:
                # 进入batch钩子上下文
                with self.hooks.begin(
                    "on_batch", self._get_hook_args(batch=batch)
                ) as on_batch_hook_args:
                    # 执行批处理器：前向传播、损失计算、反向传播
                    batch_processor(
                        pipeline_hooks=self.hooks,          # 钩子系统
                        on_batch_hook_args=on_batch_hook_args, # 批次钩子参数
                        model=self.model,                   # 训练模型
                    )
                    
                    # 将batch钩子的输出传递给step钩子
                    # 这样step级别的钩子也能访问到模型输出和损失
                    on_step_hook_args.model_outputs = (
                        on_batch_hook_args.model_outputs
                    )
                    on_step_hook_args.reduce_loss = (
                        on_batch_hook_args.reduce_loss
                    )

        # === 主训练循环 ===
        # 进入loop钩子上下文（最外层）
        with self.hooks.begin(
            "on_loop", self._get_hook_args()
        ) as on_loop_hook_args:
            
            # 训练主循环：直到达到停止条件
            while not end_loop_flag:
                
                # TODO: 当不同进程的批次数量不同时，同步end_loop_flag！
                #
                # 在某些情况下，当数据集被分割到不同进程时，
                # 数据加载器可能没有相同数量的批次。
                #
                # 如果数据加载器的批次数量不同，
                # 训练循环可能会挂起或产生意外结果。
                #
                # 考虑使用 Accelerator.join_uneven_inputs？
                
                # 进入epoch钩子上下文

                with self.hooks.begin(
                    "on_epoch", self._get_hook_args()
                ) as on_epoch_hook_args:
                    
                    # 遍历数据加载器的每个批次
                    for _i, batch in enumerate(self.dataloader):
                        
                        # TODO: 支持 Accelerator.accumulate？
                        
                        # 执行单步训练
                        step(batch=batch, batch_processor=self.batch_processor)
                        
                        # 更新步骤计数器
                        self.trainer_progress_state.update_step()
                        
                        # 同步训练状态到epoch钩子参数
                        self.trainer_progress_state.sync_pipeline_hook_arg(
                            on_epoch_hook_args
                        )
                        
                        # 检查是否达到停止条件（按步数）
                        if self.trainer_progress_state.is_training_end(
                            max_step=self.max_step, max_epoch=self.max_epoch
                        ):
                            end_loop_flag = True
                            break  # 退出批次循环

                # 更新轮次计数器
                self.trainer_progress_state.update_epoch()
                
                # 同步训练状态到loop钩子参数
                self.trainer_progress_state.sync_pipeline_hook_arg(
                    on_loop_hook_args
                )
                
                # 检查是否达到停止条件（按轮次）
                if self.trainer_progress_state.is_training_end(
                    max_step=self.max_step, max_epoch=self.max_epoch
                ):
                    end_loop_flag = True

        # === 训练结束日志 ===
        
        logger.info(
            "\n" + "=" * 50 + "FINISH TRAINING" + "=" * 50,
            main_process_only=True,
        )