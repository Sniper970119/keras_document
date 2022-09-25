# keras.engine.base_layer

## def disable_tracking()

一个装饰器，用来关闭状态跟踪。

## class keras.engine.base_layer.Layer()

| 参数                      | 说明                     |
|:------------------------|------------------------|
| input, output           | 输入和输出                  |
| input_mask, output_mask | 输入和输出的mask             |
| input_shape             | 输入的维度                  |
| input_spec              | 输入信息的描述，用来指定一些关键信息给使用者 |
| name                    | 层的名字                   |
| non_trainable_weights   | 不可训练参数的权重              |
| output_shape            | 输出的维度                  |
| stateful                | 是否有weight状态（RNNs需要）    |
| supports_masking        | 是否支持mask               |
| trainable               | 当前层是否可训练               |
| trainable_weights       | 可以训练的权重列表              |
| uses_learning_phase     | 当前层处于测试阶段还是训练阶段        |
| weights                 | 当前层的值                  |
| dtype                   | 数据类型                   |

keras所有层、Metric等的基类。

### def \_\_init__()

初始化类，在这里会初始化一些私有变量，同时对例如层的名字、层是否可以训练进行基础初始化。

### def \_\_node_key__()

根据传入的layer name 和 在[Node](#Todo)中的index构建独特的id标识。主要是`Network()._network_nodes` 对layer进行一一对应。
[Network](#Todo)是layers组成的有向无环图，Model只是添加了训练方法的Network。

### def losses()

(#Todo)

### def updates()

(#Todo)

### def built()

模型是否被构建的标志，默认为False，当模型被self.build后会置为True，防止weight的重复初始化。
(这个可以在`Layer().__call__`中看到`if not self.built:`)

### def trainable_weights()

首先会在一开始判断当前层是否为trainable，并返回可训练权重集合。

### def non_trainable_weights()

不可训练权重。它和trainable_weights()一样，都可以进行update()。（不可训练参数的update例如LayerNorm）

### def add_weight()

装填参数，包括可训练参数和不可训练参数。

### def assert_input_compatibility()

检查当前的输入是否匹配当前层。

对于输入类型检查，首先判断所有的输入是否都是keras_tensor(通过是否含有属性_keras_history)。

而对于输入维度检查，因为self.input_spec主要负责"输入信息的描述，用来指定一些关键信息给使用者"，因此也用来在这里进行维度检查。 如果没有输入input_spec，则不进行维度检查。

随后遍历输入inputs和input_spec，进行检查：

        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):

首先检查ndim，包括ndim、max_ndim、min_ndim:

    # Check ndim.
    if spec.ndim is not None:
        if K.ndim(x) != spec.ndim:
            raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
    if spec.max_ndim is not None:
        ndim = K.ndim(x)
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
    if spec.min_ndim is not None:
        ndim = K.ndim(x)
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))

随后，类似的方法分别检查了dtype、axes以及shape。

### def call()

当前层的计算逻辑定义的地方。

这里返回了inputs,其实raise NotImplementedError 也可以，因为需要子类详细定义call()。

### def \_\_call__()

将该类变为可直接调用的形式，同时包装了一下def call()。

在这里，首先会检查当前层是否build过(self.built is False)，如果没有，则先build该层。

随后后获取当前层输入的mask信息，调用self.call()、调用self.compute_mask()，随后将输入和输出以及其他相关信息添加到Node中去 （self._add_inbound_node）。

### def _add_inbound_node()

创建Node节点。

### def compute_output_shape()

计算输出shape。输入input_shape，需要根据层的具体逻辑输出output_shape。默认返回input_shape。

### def compute_mask()

计算mask。首先会判断self.supports_masking，随后检查mask值，通过检查则直接返回mask。

如果当前层支持mask(self.supports_masking)，mask的值会从前置层传过来。[](#Todo)

### def metrics()

返回当前类的metrics

## class keras.engine.base_layer.InputSpec()

| 参数       | 说明                |
|:---------|-------------------|
| dtype    | 输入的数据类型           |
| shape    | 输入数据的shape        |
| ndim     | 输入数据的维度           |
| max_ndim | 最大输入维度            |
| min_ndim | 最小输入维度            |
| axes     | 合法的tensor.shape集合 |

该类用来对当前层进行一些描述，方便该层的使用者快速的了解一些基本输入输出格式。

### def \_\_repr__()

详细的定义该类的print(  )样式。

## class keras.engine.base_layer.Node()

相关文章：[Layer、Node与Tensor之间的关系](http://www.sniper97.cn/index.php/note/deep-learning/note-deep-learning/4286/)

| 参数              | 说明                               |
|:----------------|----------------------------------|
| outbound_layer  | 当前Node所属的Layer                   |
| inbound_layers  | 输入到当前Node的Layer                  |
| node_indices    | Layer中当前Node的索引                  |
| tensor_indices  | Layer中当前output_tensors中tensor的索引 |
| input_tensors   | 当前Node的输入tensor                  |
| output_tensors  | 当前Node的输出tensor                  |
| input_masks     | 输入mask                           |
| output_masks    | 输出mask                           |
| input_shapes    | input_tensors的维度                 |
| output_shapes   | output_tensors的维度                |
| arguments       | 其他参数                             |

### \_\_init__()

首先是一些变量的直接赋值。

单独一提的是，在Layer类中，`_add_inbound_node` 中初始化Node时的代码是这样的

        Node(
            self,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )

而Node中的init函数的定义是这样的

    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes,
                 arguments=None)

可以看到，Node的第一个参数也就是`outbound_layer`传入的是Layer类本身。因此`outbound_layer`就是Node所属的这个Layer， 在Node初始化时，就已经绑定了所属的Layer。

而其他参数已经在上文中给到了较为详细的解释。

但在代码的最后有一段

        # Add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer._outbound_nodes.append(self)
        outbound_layer._inbound_nodes.append(self)

这段代码将`inbound_layers`中的每一个输入层的`outbound_nodes`中添加当前Node。同时向当前层的`inbound_nodes`添加当前Node。 通过这种方式，将当前层与前置层连接起来。

### def get_config()

显示的输出当前Node的一些信息。