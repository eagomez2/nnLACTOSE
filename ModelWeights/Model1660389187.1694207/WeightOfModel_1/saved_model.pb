��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.9.12v2.9.0-18-gd8ce9f9c3018Ә
t
gru/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegru/Variable
m
 gru/Variable/Read/ReadVariableOpReadVariableOpgru/Variable*
_output_shapes

:*
dtype0
z
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namegru/gru_cell/bias
s
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:*
dtype0
�
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namegru/gru_cell/recurrent_kernel
�
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

:*
dtype0
�
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namegru/gru_cell/kernel
{
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�+B�+ B�+
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
C
0
1
62
73
84
&5
'6
.7
/8*
C
0
1
62
73
84
&5
'6
.7
/8*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
6
Btrace_0
Ctrace_1
Dtrace_2
Etrace_3* 
* 

Fserving_default* 

0
1*

0
1*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71
82*

60
71
82*
* 
�

Nstates
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
* 
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator

6kernel
7recurrent_kernel
8bias*
* 

&0
'1*

&0
'1*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

vtrace_0* 

wtrace_0* 
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

x0*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

60
71
82*

60
71
82*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
8
~trace_0
trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
hb
VARIABLE_VALUEgru/VariableBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
n
serving_default_InputPlaceholder*"
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputdense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/biasgru/gru_cell/recurrent_kernelgru/Variabledense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_32766323
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOp gru/Variable/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_32767922
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasgru/Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_32767962��
�
�
while_cond_32766016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32766016___redundant_placeholder06
2while_while_cond_32766016___redundant_placeholder16
2while_while_cond_32766016___redundant_placeholder26
2while_while_cond_32766016___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�D
�
while_body_32766961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�W
�
A__inference_gru_layer_call_and_return_conditional_losses_32767383

inputs9
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          d
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:Z
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32767287*
condR
while_cond_32767286*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Y
IdentityIdentitytranspose_1:y:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
while_cond_32765714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32765714___redundant_placeholder06
2while_while_cond_32765714___redundant_placeholder16
2while_while_cond_32765714___redundant_placeholder26
2while_while_cond_32765714___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�D
�
while_body_32767287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
� 
�
!__inference__traced_save_32767922
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop+
'savev2_gru_variable_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop'savev2_gru_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*k
_input_shapesZ
X: ::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

::

_output_shapes
: 
�$
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767773

inputs
states_00
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask]
MatMul_1MatMulstates_0strided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:L
mulMulSigmoid_1:y:0states_0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:L
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:F B

_output_shapes

:
 
_user_specified_nameinputs:HD

_output_shapes

:
"
_user_specified_name
states/0
�
�
while_cond_32767123
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32767123___redundant_placeholder06
2while_while_cond_32767123___redundant_placeholder16
2while_while_cond_32767123___redundant_placeholder26
2while_while_cond_32767123___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�
�
&__inference_gru_layer_call_fn_32766894

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32766113j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�D
�
while_body_32767124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�(
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767727

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�mul/ReadVariableOp�mul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskX
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0u
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0strided_slice:output:0*
T0*
_output_shapes
:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_splitQ
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:>
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:S
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:B
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:S
mul/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0X
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask_
MatMul_2BatchMatMulV2mul:z:0strided_slice_1:output:0*
T0*
_output_shapes
:T
add_2AddV2split:output:2MatMul_2:output:0*
T0*
_output_shapes
::
TanhTanh	add_2:z:0*
T0*
_output_shapes
:U
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0Z
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:B
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:G
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:I
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:K

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�$
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765296

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask[
MatMul_1MatMulstatesstrided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:J
mulMulSigmoid_1:y:0states*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:J
mul_1MulSigmoid:y:0states*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:F B

_output_shapes

:
 
_user_specified_nameinputs:FB

_output_shapes

:
 
_user_specified_namestates
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_32765883

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
(__inference_model_layer_call_fn_32766348

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_32765886f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
F
*__inference_flatten_layer_call_fn_32767613

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_32765883W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�J
�
gru_while_body_32766677$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0E
3gru_while_gru_cell_matmul_readvariableop_resource_0:B
4gru_while_gru_cell_biasadd_readvariableop_resource_0:>
,gru_while_gru_cell_readvariableop_resource_0:
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorC
1gru_while_gru_cell_matmul_readvariableop_resource:@
2gru_while_gru_cell_biasadd_readvariableop_resource:<
*gru_while_gru_cell_readvariableop_resource:��)gru/while/gru_cell/BiasAdd/ReadVariableOp�(gru/while/gru_cell/MatMul/ReadVariableOp�!gru/while/gru_cell/ReadVariableOp�#gru/while/gru_cell/ReadVariableOp_1�
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)gru/while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp4gru_while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:01gru/while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:m
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 gru/while/gru_cell/strided_sliceStridedSlice)gru/while/gru_cell/ReadVariableOp:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_2)gru/while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:m
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/MatMul_1:product:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*
_output_shapes

:j
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*
_output_shapes

:n
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0gru_while_placeholder_2*
T0*
_output_shapes

:�
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2%gru/while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:f
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*
_output_shapes

:]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: �
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: x
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*
_output_shapes

:�
gru/while/NoOpNoOp*^gru/while/gru_cell/BiasAdd/ReadVariableOp)^gru/while/gru_cell/MatMul/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "j
2gru_while_gru_cell_biasadd_readvariableop_resource4gru_while_gru_cell_biasadd_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"�
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2V
)gru/while/gru_cell/BiasAdd/ReadVariableOp)gru/while/gru_cell/BiasAdd/ReadVariableOp2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�D
�
while_body_32765715
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�
�
while_body_32765229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_32765297_0:'
while_gru_cell_32765299_0:+
while_gru_cell_32765301_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_32765297:%
while_gru_cell_32765299:)
while_gru_cell_32765301:��&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_32765297_0while_gru_cell_32765299_0while_gru_cell_32765301_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765296�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*
_output_shapes

:u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_32765297while_gru_cell_32765297_0"4
while_gru_cell_32765299while_gru_cell_32765299_0"4
while_gru_cell_32765301while_gru_cell_32765301_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�	
�
+__inference_gru_cell_layer_call_fn_32767633

inputs
states_0
unknown:
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765296f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:h

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:HD

_output_shapes

:
"
_user_specified_name
states/0
�$
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767819

inputs
states_00
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask]
MatMul_1MatMulstates_0strided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:L
mulMulSigmoid_1:y:0states_0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:L
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:F B

_output_shapes

:
 
_user_specified_nameinputs:HD

_output_shapes

:
"
_user_specified_name
states/0
�$
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765414

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask[
MatMul_1MatMulstatesstrided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:J
mulMulSigmoid_1:y:0states*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:J
mul_1MulSigmoid:y:0states*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:F B

_output_shapes

:
 
_user_specified_nameinputs:FB

_output_shapes

:
 
_user_specified_namestates
�W
�
A__inference_gru_layer_call_and_return_conditional_losses_32765811

inputs9
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          d
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:Z
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32765715*
condR
while_cond_32765714*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Y
IdentityIdentitytranspose_1:y:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�X
�
A__inference_gru_layer_call_and_return_conditional_losses_32767220
inputs_09
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:���������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32767124*
condR
while_cond_32767123*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0
�
�
*__inference_dense_2_layer_call_fn_32767586

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
while_cond_32767286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32767286___redundant_placeholder06
2while_while_cond_32767286___redundant_placeholder16
2while_while_cond_32767286___redundant_placeholder26
2while_while_cond_32767286___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�	
�
model_gru_while_cond_327650070
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_20
,model_gru_while_less_model_gru_strided_sliceJ
Fmodel_gru_while_model_gru_while_cond_32765007___redundant_placeholder0J
Fmodel_gru_while_model_gru_while_cond_32765007___redundant_placeholder1J
Fmodel_gru_while_model_gru_while_cond_32765007___redundant_placeholder2J
Fmodel_gru_while_model_gru_while_cond_32765007___redundant_placeholder3
model_gru_while_identity
�
model/gru/while/LessLessmodel_gru_while_placeholder,model_gru_while_less_model_gru_strided_slice*
T0*
_output_shapes
: _
model/gru/while/IdentityIdentitymodel/gru/while/Less:z:0*
T0
*
_output_shapes
: "=
model_gru_while_identity!model/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�X
�
A__inference_gru_layer_call_and_return_conditional_losses_32767057
inputs_09
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:���������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32766961*
condR
while_cond_32766960*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0
�/
�
A__inference_gru_layer_call_and_return_conditional_losses_32765339

inputs#
gru_cell_32765214:#
gru_cell_32765216:
gru_cell_32765218:#
gru_cell_32765220:
identity��AssignVariableOp�ReadVariableOp� gru_cell/StatefulPartitionedCall�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_32765214gru_cell_32765216gru_cell_32765218gru_cell_32765220*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765213n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : `
ReadVariableOpReadVariableOpgru_cell_32765214*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_32765216gru_cell_32765218gru_cell_32765220*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32765229*
condR
while_cond_32765228*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOpgru_cell_32765214while:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype0*
validate_shape(b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_32767577

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�

#__inference__wrapped_model_32765142	
input?
-model_dense_tensordot_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:C
1model_gru_gru_cell_matmul_readvariableop_resource:@
2model_gru_gru_cell_biasadd_readvariableop_resource:<
*model_gru_gru_cell_readvariableop_resource:E
3model_gru_gru_cell_matmul_1_readvariableop_resource:A
/model_dense_1_tensordot_readvariableop_resource:;
-model_dense_1_biasadd_readvariableop_resource:A
/model_dense_2_tensordot_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�$model/dense/Tensordot/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�&model/dense_1/Tensordot/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�&model/dense_2/Tensordot/ReadVariableOp�model/gru/AssignVariableOp�model/gru/ReadVariableOp�)model/gru/gru_cell/BiasAdd/ReadVariableOp�(model/gru/gru_cell/MatMul/ReadVariableOp�*model/gru/gru_cell/MatMul_1/ReadVariableOp�!model/gru/gru_cell/ReadVariableOp�#model/gru/gru_cell/ReadVariableOp_1�%model/gru/gru_cell/mul/ReadVariableOp�'model/gru/gru_cell/mul_1/ReadVariableOp�model/gru/while�
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0t
#model/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
model/dense/Tensordot/ReshapeReshapeinput,model/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:p
model/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0$model/dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:[
model/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/dense/Gelu/mulMulmodel/dense/Gelu/mul/x:output:0model/dense/BiasAdd:output:0*
T0*"
_output_shapes
:\
model/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
model/dense/Gelu/truedivRealDivmodel/dense/BiasAdd:output:0 model/dense/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:f
model/dense/Gelu/ErfErfmodel/dense/Gelu/truediv:z:0*
T0*"
_output_shapes
:[
model/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense/Gelu/addAddV2model/dense/Gelu/add/x:output:0model/dense/Gelu/Erf:y:0*
T0*"
_output_shapes
:~
model/dense/Gelu/mul_1Mulmodel/dense/Gelu/mul:z:0model/dense/Gelu/add:z:0*
T0*"
_output_shapes
:m
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/gru/transpose	Transposemodel/dense/Gelu/mul_1:z:0!model/gru/transpose/perm:output:0*
T0*"
_output_shapes
:d
model/gru/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         g
model/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
model/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
model/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
%model/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0 model/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
model/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_slice_1StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_1/stack:output:0*model/gru/strided_slice_1/stack_1:output:0*model/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
(model/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp1model_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model/gru/gru_cell/MatMulMatMul"model/gru/strided_slice_1:output:00model/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)model/gru/gru_cell/BiasAdd/ReadVariableOpReadVariableOp2model_gru_gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/gru/gru_cell/BiasAddBiasAdd#model/gru/gru_cell/MatMul:product:01model/gru/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:m
"model/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/gru_cell/splitSplit+model/gru/gru_cell/split/split_dim:output:0#model/gru/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
!model/gru/gru_cell/ReadVariableOpReadVariableOp*model_gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0w
&model/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(model/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(model/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 model/gru/gru_cell/strided_sliceStridedSlice)model/gru/gru_cell/ReadVariableOp:value:0/model/gru/gru_cell/strided_slice/stack:output:01model/gru/gru_cell/strided_slice/stack_1:output:01model/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
*model/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp3model_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
model/gru/gru_cell/MatMul_1MatMul2model/gru/gru_cell/MatMul_1/ReadVariableOp:value:0)model/gru/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:m
model/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$model/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/gru_cell/split_1SplitV%model/gru/gru_cell/MatMul_1:product:0!model/gru/gru_cell/Const:output:0-model/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
model/gru/gru_cell/addAddV2!model/gru/gru_cell/split:output:0#model/gru/gru_cell/split_1:output:0*
T0*
_output_shapes

:j
model/gru/gru_cell/SigmoidSigmoidmodel/gru/gru_cell/add:z:0*
T0*
_output_shapes

:�
model/gru/gru_cell/add_1AddV2!model/gru/gru_cell/split:output:1#model/gru/gru_cell/split_1:output:1*
T0*
_output_shapes

:n
model/gru/gru_cell/Sigmoid_1Sigmoidmodel/gru/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
%model/gru/gru_cell/mul/ReadVariableOpReadVariableOp3model_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
model/gru/gru_cell/mulMul model/gru/gru_cell/Sigmoid_1:y:0-model/gru/gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
#model/gru/gru_cell/ReadVariableOp_1ReadVariableOp*model_gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0y
(model/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*model/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*model/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
"model/gru/gru_cell/strided_slice_1StridedSlice+model/gru/gru_cell/ReadVariableOp_1:value:01model/gru/gru_cell/strided_slice_1/stack:output:03model/gru/gru_cell/strided_slice_1/stack_1:output:03model/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
model/gru/gru_cell/MatMul_2MatMulmodel/gru/gru_cell/mul:z:0+model/gru/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
model/gru/gru_cell/add_2AddV2!model/gru/gru_cell/split:output:2%model/gru/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:f
model/gru/gru_cell/TanhTanhmodel/gru/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
'model/gru/gru_cell/mul_1/ReadVariableOpReadVariableOp3model_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
model/gru/gru_cell/mul_1Mulmodel/gru/gru_cell/Sigmoid:y:0/model/gru/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:]
model/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/gru/gru_cell/subSub!model/gru/gru_cell/sub/x:output:0model/gru/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:�
model/gru/gru_cell/mul_2Mulmodel/gru/gru_cell/sub:z:0model/gru/gru_cell/Tanh:y:0*
T0*
_output_shapes

:�
model/gru/gru_cell/add_3AddV2model/gru/gru_cell/mul_1:z:0model/gru/gru_cell/mul_2:z:0*
T0*
_output_shapes

:x
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0 model/gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���P
model/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : �
model/gru/ReadVariableOpReadVariableOp3model_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
"model/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0 model/gru/ReadVariableOp:value:0 model/gru/strided_slice:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:01model_gru_gru_cell_matmul_readvariableop_resource2model_gru_gru_cell_biasadd_readvariableop_resource*model_gru_gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *)
body!R
model_gru_while_body_32765008*)
cond!R
model_gru_while_cond_32765007*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0r
model/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������k
!model/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_slice_2StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masko
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/gru/transpose_1	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_1/perm:output:0*
T0*"
_output_shapes
:e
model/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
model/gru/AssignVariableOpAssignVariableOp3model_gru_gru_cell_matmul_1_readvariableop_resourcemodel/gru/while:output:4^model/gru/ReadVariableOp+^model/gru/gru_cell/MatMul_1/ReadVariableOp&^model/gru/gru_cell/mul/ReadVariableOp(^model/gru/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&model/dense_1/Tensordot/ReadVariableOpReadVariableOp/model_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
%model/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
model/dense_1/Tensordot/ReshapeReshapemodel/gru/transpose_1:y:0.model/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
model/dense_1/Tensordot/MatMulMatMul(model/dense_1/Tensordot/Reshape:output:0.model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:r
model/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
model/dense_1/TensordotReshape(model/dense_1/Tensordot/MatMul:product:0&model/dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_1/BiasAddBiasAdd model/dense_1/Tensordot:output:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:]
model/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/dense_1/Gelu/mulMul!model/dense_1/Gelu/mul/x:output:0model/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:^
model/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
model/dense_1/Gelu/truedivRealDivmodel/dense_1/BiasAdd:output:0"model/dense_1/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:j
model/dense_1/Gelu/ErfErfmodel/dense_1/Gelu/truediv:z:0*
T0*"
_output_shapes
:]
model/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_1/Gelu/addAddV2!model/dense_1/Gelu/add/x:output:0model/dense_1/Gelu/Erf:y:0*
T0*"
_output_shapes
:�
model/dense_1/Gelu/mul_1Mulmodel/dense_1/Gelu/mul:z:0model/dense_1/Gelu/add:z:0*
T0*"
_output_shapes
:�
&model/dense_2/Tensordot/ReadVariableOpReadVariableOp/model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
%model/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
model/dense_2/Tensordot/ReshapeReshapemodel/dense_1/Gelu/mul_1:z:0.model/dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
model/dense_2/Tensordot/MatMulMatMul(model/dense_2/Tensordot/Reshape:output:0.model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:r
model/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
model/dense_2/TensordotReshape(model/dense_2/Tensordot/MatMul:product:0&model/dense_2/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_2/BiasAddBiasAdd model/dense_2/Tensordot:output:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:]
model/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/dense_2/Gelu/mulMul!model/dense_2/Gelu/mul/x:output:0model/dense_2/BiasAdd:output:0*
T0*"
_output_shapes
:^
model/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
model/dense_2/Gelu/truedivRealDivmodel/dense_2/BiasAdd:output:0"model/dense_2/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:j
model/dense_2/Gelu/ErfErfmodel/dense_2/Gelu/truediv:z:0*
T0*"
_output_shapes
:]
model/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_2/Gelu/addAddV2!model/dense_2/Gelu/add/x:output:0model/dense_2/Gelu/Erf:y:0*
T0*"
_output_shapes
:�
model/dense_2/Gelu/mul_1Mulmodel/dense_2/Gelu/mul:z:0model/dense_2/Gelu/add:z:0*
T0*"
_output_shapes
:d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshapemodel/dense_2/Gelu/mul_1:z:0model/flatten/Const:output:0*
T0*
_output_shapes

:d
IdentityIdentitymodel/flatten/Reshape:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp'^model/dense_1/Tensordot/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp'^model/dense_2/Tensordot/ReadVariableOp^model/gru/AssignVariableOp^model/gru/ReadVariableOp*^model/gru/gru_cell/BiasAdd/ReadVariableOp)^model/gru/gru_cell/MatMul/ReadVariableOp+^model/gru/gru_cell/MatMul_1/ReadVariableOp"^model/gru/gru_cell/ReadVariableOp$^model/gru/gru_cell/ReadVariableOp_1&^model/gru/gru_cell/mul/ReadVariableOp(^model/gru/gru_cell/mul_1/ReadVariableOp^model/gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2P
&model/dense_1/Tensordot/ReadVariableOp&model/dense_1/Tensordot/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2P
&model/dense_2/Tensordot/ReadVariableOp&model/dense_2/Tensordot/ReadVariableOp28
model/gru/AssignVariableOpmodel/gru/AssignVariableOp24
model/gru/ReadVariableOpmodel/gru/ReadVariableOp2V
)model/gru/gru_cell/BiasAdd/ReadVariableOp)model/gru/gru_cell/BiasAdd/ReadVariableOp2T
(model/gru/gru_cell/MatMul/ReadVariableOp(model/gru/gru_cell/MatMul/ReadVariableOp2X
*model/gru/gru_cell/MatMul_1/ReadVariableOp*model/gru/gru_cell/MatMul_1/ReadVariableOp2F
!model/gru/gru_cell/ReadVariableOp!model/gru/gru_cell/ReadVariableOp2J
#model/gru/gru_cell/ReadVariableOp_1#model/gru/gru_cell/ReadVariableOp_12N
%model/gru/gru_cell/mul/ReadVariableOp%model/gru/gru_cell/mul/ReadVariableOp2R
'model/gru/gru_cell/mul_1/ReadVariableOp'model/gru/gru_cell/mul_1/ReadVariableOp2"
model/gru/whilemodel/gru/while:I E
"
_output_shapes
:

_user_specified_nameInput
�
�
while_cond_32766960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32766960___redundant_placeholder06
2while_while_cond_32766960___redundant_placeholder16
2while_while_cond_32766960___redundant_placeholder26
2while_while_cond_32766960___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_32767619

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
gru_while_cond_32766676$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice>
:gru_while_gru_while_cond_32766676___redundant_placeholder0>
:gru_while_gru_while_cond_32766676___redundant_placeholder1>
:gru_while_gru_while_cond_32766676___redundant_placeholder2>
:gru_while_gru_while_cond_32766676___redundant_placeholder3
gru_while_identity
p
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�
�
&__inference_gru_layer_call_fn_32766881

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32765811j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�/
�
A__inference_gru_layer_call_and_return_conditional_losses_32765603

inputs#
gru_cell_32765524:#
gru_cell_32765526:
gru_cell_32765528:#
gru_cell_32765530:
identity��AssignVariableOp�ReadVariableOp� gru_cell/StatefulPartitionedCall�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_32765524gru_cell_32765526gru_cell_32765528gru_cell_32765530*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765492n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : `
ReadVariableOpReadVariableOpgru_cell_32765524*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_32765526gru_cell_32765528gru_cell_32765530*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32765539*
condR
while_cond_32765538*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOpgru_cell_32765524while:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype0*
validate_shape(b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�W
�
A__inference_gru_layer_call_and_return_conditional_losses_32767546

inputs9
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          d
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:Z
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32767450*
condR
while_cond_32767449*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Y
IdentityIdentitytranspose_1:y:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
*__inference_dense_1_layer_call_fn_32767555

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
while_body_32765539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_32765561_0:'
while_gru_cell_32765563_0:+
while_gru_cell_32765565_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_32765561:%
while_gru_cell_32765563:)
while_gru_cell_32765565:��&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_32765561_0while_gru_cell_32765563_0while_gru_cell_32765565_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765414�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*
_output_shapes

:u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_32765561while_gru_cell_32765561_0"4
while_gru_cell_32765563while_gru_cell_32765563_0"4
while_gru_cell_32765565while_gru_cell_32765565_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�R
�

model_gru_while_body_327650080
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_2-
)model_gru_while_model_gru_strided_slice_0k
gmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0K
9model_gru_while_gru_cell_matmul_readvariableop_resource_0:H
:model_gru_while_gru_cell_biasadd_readvariableop_resource_0:D
2model_gru_while_gru_cell_readvariableop_resource_0:
model_gru_while_identity
model_gru_while_identity_1
model_gru_while_identity_2
model_gru_while_identity_3
model_gru_while_identity_4+
'model_gru_while_model_gru_strided_slicei
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorI
7model_gru_while_gru_cell_matmul_readvariableop_resource:F
8model_gru_while_gru_cell_biasadd_readvariableop_resource:B
0model_gru_while_gru_cell_readvariableop_resource:��/model/gru/while/gru_cell/BiasAdd/ReadVariableOp�.model/gru/while/gru_cell/MatMul/ReadVariableOp�'model/gru/while/gru_cell/ReadVariableOp�)model/gru/while/gru_cell/ReadVariableOp_1�
Amodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
3model/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0model_gru_while_placeholderJmodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
.model/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp9model_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
model/gru/while/gru_cell/MatMulMatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:06model/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
/model/gru/while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp:model_gru_while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
 model/gru/while/gru_cell/BiasAddBiasAdd)model/gru/while/gru_cell/MatMul:product:07model/gru/while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:s
(model/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/while/gru_cell/splitSplit1model/gru/while/gru_cell/split/split_dim:output:0)model/gru/while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
'model/gru/while/gru_cell/ReadVariableOpReadVariableOp2model_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0}
,model/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.model/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.model/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
&model/gru/while/gru_cell/strided_sliceStridedSlice/model/gru/while/gru_cell/ReadVariableOp:value:05model/gru/while/gru_cell/strided_slice/stack:output:07model/gru/while/gru_cell/strided_slice/stack_1:output:07model/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
!model/gru/while/gru_cell/MatMul_1MatMulmodel_gru_while_placeholder_2/model/gru/while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:s
model/gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����u
*model/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/gru/while/gru_cell/split_1SplitV+model/gru/while/gru_cell/MatMul_1:product:0'model/gru/while/gru_cell/Const:output:03model/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
model/gru/while/gru_cell/addAddV2'model/gru/while/gru_cell/split:output:0)model/gru/while/gru_cell/split_1:output:0*
T0*
_output_shapes

:v
 model/gru/while/gru_cell/SigmoidSigmoid model/gru/while/gru_cell/add:z:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/add_1AddV2'model/gru/while/gru_cell/split:output:1)model/gru/while/gru_cell/split_1:output:1*
T0*
_output_shapes

:z
"model/gru/while/gru_cell/Sigmoid_1Sigmoid"model/gru/while/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/mulMul&model/gru/while/gru_cell/Sigmoid_1:y:0model_gru_while_placeholder_2*
T0*
_output_shapes

:�
)model/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp2model_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0
.model/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
0model/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
0model/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
(model/gru/while/gru_cell/strided_slice_1StridedSlice1model/gru/while/gru_cell/ReadVariableOp_1:value:07model/gru/while/gru_cell/strided_slice_1/stack:output:09model/gru/while/gru_cell/strided_slice_1/stack_1:output:09model/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
!model/gru/while/gru_cell/MatMul_2MatMul model/gru/while/gru_cell/mul:z:01model/gru/while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/add_2AddV2'model/gru/while/gru_cell/split:output:2+model/gru/while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:r
model/gru/while/gru_cell/TanhTanh"model/gru/while/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/mul_1Mul$model/gru/while/gru_cell/Sigmoid:y:0model_gru_while_placeholder_2*
T0*
_output_shapes

:c
model/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/gru/while/gru_cell/subSub'model/gru/while/gru_cell/sub/x:output:0$model/gru/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/mul_2Mul model/gru/while/gru_cell/sub:z:0!model/gru/while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:�
model/gru/while/gru_cell/add_3AddV2"model/gru/while/gru_cell/mul_1:z:0"model/gru/while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
4model/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_while_placeholder_1model_gru_while_placeholder"model/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���W
model/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :z
model/gru/while/addAddV2model_gru_while_placeholdermodel/gru/while/add/y:output:0*
T0*
_output_shapes
: Y
model/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/gru/while/add_1AddV2,model_gru_while_model_gru_while_loop_counter model/gru/while/add_1/y:output:0*
T0*
_output_shapes
: w
model/gru/while/IdentityIdentitymodel/gru/while/add_1:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: �
model/gru/while/Identity_1Identity2model_gru_while_model_gru_while_maximum_iterations^model/gru/while/NoOp*
T0*
_output_shapes
: w
model/gru/while/Identity_2Identitymodel/gru/while/add:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: �
model/gru/while/Identity_3IdentityDmodel/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/gru/while/NoOp*
T0*
_output_shapes
: �
model/gru/while/Identity_4Identity"model/gru/while/gru_cell/add_3:z:0^model/gru/while/NoOp*
T0*
_output_shapes

:�
model/gru/while/NoOpNoOp0^model/gru/while/gru_cell/BiasAdd/ReadVariableOp/^model/gru/while/gru_cell/MatMul/ReadVariableOp(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "v
8model_gru_while_gru_cell_biasadd_readvariableop_resource:model_gru_while_gru_cell_biasadd_readvariableop_resource_0"t
7model_gru_while_gru_cell_matmul_readvariableop_resource9model_gru_while_gru_cell_matmul_readvariableop_resource_0"f
0model_gru_while_gru_cell_readvariableop_resource2model_gru_while_gru_cell_readvariableop_resource_0"=
model_gru_while_identity!model/gru/while/Identity:output:0"A
model_gru_while_identity_1#model/gru/while/Identity_1:output:0"A
model_gru_while_identity_2#model/gru/while/Identity_2:output:0"A
model_gru_while_identity_3#model/gru/while/Identity_3:output:0"A
model_gru_while_identity_4#model/gru/while/Identity_4:output:0"T
'model_gru_while_model_gru_strided_slice)model_gru_while_model_gru_strided_slice_0"�
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2b
/model/gru/while/gru_cell/BiasAdd/ReadVariableOp/model/gru/while/gru_cell/BiasAdd/ReadVariableOp2`
.model/gru/while/gru_cell/MatMul/ReadVariableOp.model/gru/while/gru_cell/MatMul/ReadVariableOp2R
'model/gru/while/gru_cell/ReadVariableOp'model/gru/while/gru_cell/ReadVariableOp2V
)model/gru/while/gru_cell/ReadVariableOp_1)model/gru/while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_32765228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32765228___redundant_placeholder06
2while_while_cond_32765228___redundant_placeholder16
2while_while_cond_32765228___redundant_placeholder26
2while_while_cond_32765228___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
��
�	
C__inference_model_layer_call_and_return_conditional_losses_32766592

inputs9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:=
+gru_gru_cell_matmul_readvariableop_resource::
,gru_gru_cell_biasadd_readvariableop_resource:6
$gru_gru_cell_readvariableop_resource:?
-gru_gru_cell_matmul_1_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�gru/AssignVariableOp�gru/ReadVariableOp�#gru/gru_cell/BiasAdd/ReadVariableOp�"gru/gru_cell/MatMul/ReadVariableOp�$gru/gru_cell/MatMul_1/ReadVariableOp�gru/gru_cell/ReadVariableOp�gru/gru_cell/ReadVariableOp_1�gru/gru_cell/mul/ReadVariableOp�!gru/gru_cell/mul_1/ReadVariableOp�	gru/while�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:j
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:U
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*"
_output_shapes
:V
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?~
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:Z
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*"
_output_shapes
:U
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*"
_output_shapes
:l
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*"
_output_shapes
:g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
gru/transpose	Transposedense/Gelu/mul_1:z:0gru/transpose/perm:output:0*
T0*"
_output_shapes
:^
	gru/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
#gru/gru_cell/BiasAdd/ReadVariableOpReadVariableOp,gru_gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0+gru/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:g
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0q
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru/gru_cell/strided_sliceStridedSlice#gru/gru_cell/ReadVariableOp:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/MatMul_1MatMul,gru/gru_cell/MatMul_1/ReadVariableOp:value:0#gru/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:g
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����i
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/gru_cell/split_1SplitVgru/gru_cell/MatMul_1:product:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split~
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*
_output_shapes

:^
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*
_output_shapes

:�
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*
_output_shapes

:b
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru/gru_cell/mul/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0'gru/gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru/gru_cell/ReadVariableOp_1ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0s
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_1:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:Z
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
!gru/gru_cell/mul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0)gru/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:W
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:o
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*
_output_shapes

:t
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*
_output_shapes

:r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : �
gru/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/ReadVariableOp:value:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_gru_cell_matmul_readvariableop_resource,gru_gru_cell_biasadd_readvariableop_resource$gru_gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_while_body_32766458*#
condR
gru_while_cond_32766457*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0l
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*"
_output_shapes
:_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
gru/AssignVariableOpAssignVariableOp-gru_gru_cell_matmul_1_readvariableop_resourcegru/while:output:4^gru/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp ^gru/gru_cell/mul/ReadVariableOp"^gru/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
dense_1/Tensordot/ReshapeReshapegru/transpose_1:y:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:l
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:W
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?{
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:X
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:^
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*"
_output_shapes
:W
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*"
_output_shapes
:r
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*"
_output_shapes
:�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
dense_2/Tensordot/ReshapeReshapedense_1/Gelu/mul_1:z:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:l
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:W
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?{
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*"
_output_shapes
:X
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:^
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*"
_output_shapes
:W
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*"
_output_shapes
:r
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*"
_output_shapes
:^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   s
flatten/ReshapeReshapedense_2/Gelu/mul_1:z:0flatten/Const:output:0*
T0*
_output_shapes

:^
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^gru/AssignVariableOp^gru/ReadVariableOp$^gru/gru_cell/BiasAdd/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1 ^gru/gru_cell/mul/ReadVariableOp"^gru/gru_cell/mul_1/ReadVariableOp
^gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2,
gru/AssignVariableOpgru/AssignVariableOp2(
gru/ReadVariableOpgru/ReadVariableOp2J
#gru/gru_cell/BiasAdd/ReadVariableOp#gru/gru_cell/BiasAdd/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12B
gru/gru_cell/mul/ReadVariableOpgru/gru_cell/mul/ReadVariableOp2F
!gru/gru_cell/mul_1/ReadVariableOp!gru/gru_cell/mul_1/ReadVariableOp2
	gru/while	gru/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_dense_layer_call_and_return_conditional_losses_32765643

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
��
�	
C__inference_model_layer_call_and_return_conditional_losses_32766811

inputs9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:=
+gru_gru_cell_matmul_readvariableop_resource::
,gru_gru_cell_biasadd_readvariableop_resource:6
$gru_gru_cell_readvariableop_resource:?
-gru_gru_cell_matmul_1_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�gru/AssignVariableOp�gru/ReadVariableOp�#gru/gru_cell/BiasAdd/ReadVariableOp�"gru/gru_cell/MatMul/ReadVariableOp�$gru/gru_cell/MatMul_1/ReadVariableOp�gru/gru_cell/ReadVariableOp�gru/gru_cell/ReadVariableOp_1�gru/gru_cell/mul/ReadVariableOp�!gru/gru_cell/mul_1/ReadVariableOp�	gru/while�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:j
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:U
dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?u
dense/Gelu/mulMuldense/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*"
_output_shapes
:V
dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?~
dense/Gelu/truedivRealDivdense/BiasAdd:output:0dense/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:Z
dense/Gelu/ErfErfdense/Gelu/truediv:z:0*
T0*"
_output_shapes
:U
dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
dense/Gelu/addAddV2dense/Gelu/add/x:output:0dense/Gelu/Erf:y:0*
T0*"
_output_shapes
:l
dense/Gelu/mul_1Muldense/Gelu/mul:z:0dense/Gelu/add:z:0*
T0*"
_output_shapes
:g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
gru/transpose	Transposedense/Gelu/mul_1:z:0gru/transpose/perm:output:0*
T0*"
_output_shapes
:^
	gru/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_slice_1StridedSlicegru/transpose:y:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/MatMulMatMulgru/strided_slice_1:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
#gru/gru_cell/BiasAdd/ReadVariableOpReadVariableOp,gru_gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0+gru/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:g
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0q
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru/gru_cell/strided_sliceStridedSlice#gru/gru_cell/ReadVariableOp:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/MatMul_1MatMul,gru/gru_cell/MatMul_1/ReadVariableOp:value:0#gru/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:g
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����i
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/gru_cell/split_1SplitVgru/gru_cell/MatMul_1:product:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split~
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*
_output_shapes

:^
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*
_output_shapes

:�
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*
_output_shapes

:b
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru/gru_cell/mul/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0'gru/gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru/gru_cell/ReadVariableOp_1ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0s
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_1:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:Z
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
!gru/gru_cell/mul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0)gru/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:W
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:o
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*
_output_shapes

:t
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*
_output_shapes

:r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : �
gru/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/ReadVariableOp:value:0gru/strided_slice:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_gru_cell_matmul_readvariableop_resource,gru_gru_cell_biasadd_readvariableop_resource$gru_gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_while_body_32766677*#
condR
gru_while_cond_32766676*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0l
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru/strided_slice_2StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*"
_output_shapes
:_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
gru/AssignVariableOpAssignVariableOp-gru_gru_cell_matmul_1_readvariableop_resourcegru/while:output:4^gru/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp ^gru/gru_cell/mul/ReadVariableOp"^gru/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
dense_1/Tensordot/ReshapeReshapegru/transpose_1:y:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:l
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:W
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?{
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:X
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:^
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*"
_output_shapes
:W
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*"
_output_shapes
:r
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*"
_output_shapes
:�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
dense_2/Tensordot/ReshapeReshapedense_1/Gelu/mul_1:z:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:l
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:W
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?{
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*"
_output_shapes
:X
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*"
_output_shapes
:^
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*"
_output_shapes
:W
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*"
_output_shapes
:r
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*"
_output_shapes
:^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   s
flatten/ReshapeReshapedense_2/Gelu/mul_1:z:0flatten/Const:output:0*
T0*
_output_shapes

:^
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^gru/AssignVariableOp^gru/ReadVariableOp$^gru/gru_cell/BiasAdd/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1 ^gru/gru_cell/mul/ReadVariableOp"^gru/gru_cell/mul_1/ReadVariableOp
^gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2,
gru/AssignVariableOpgru/AssignVariableOp2(
gru/ReadVariableOpgru/ReadVariableOp2J
#gru/gru_cell/BiasAdd/ReadVariableOp#gru/gru_cell/BiasAdd/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12B
gru/gru_cell/mul/ReadVariableOpgru/gru_cell/mul/ReadVariableOp2F
!gru/gru_cell/mul_1/ReadVariableOp!gru/gru_cell/mul_1/ReadVariableOp2
	gru/while	gru/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
gru_while_cond_32766457$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2$
 gru_while_less_gru_strided_slice>
:gru_while_gru_while_cond_32766457___redundant_placeholder0>
:gru_while_gru_while_cond_32766457___redundant_placeholder1>
:gru_while_gru_while_cond_32766457___redundant_placeholder2>
:gru_while_gru_while_cond_32766457___redundant_placeholder3
gru_while_identity
p
gru/while/LessLessgru_while_placeholder gru_while_less_gru_strided_slice*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�D
�
while_body_32767450
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_dense_layer_call_fn_32766820

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_32765643j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
+__inference_gru_cell_layer_call_fn_32767647

inputs
states_0
unknown:
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765414f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:h

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:HD

_output_shapes

:
"
_user_specified_name
states/0
�	
�
(__inference_model_layer_call_fn_32766373

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_32766190f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
+__inference_gru_cell_layer_call_fn_32767677

inputs

states
unknown:
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765492f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:h

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�	
�
+__inference_gru_cell_layer_call_fn_32767662

inputs

states
unknown:
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765213f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:h

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�(
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767869

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�mul/ReadVariableOp�mul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskX
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0u
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0strided_slice:output:0*
T0*
_output_shapes
:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_splitQ
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:>
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:S
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:B
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:S
mul/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0X
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask_
MatMul_2BatchMatMulV2mul:z:0strided_slice_1:output:0*
T0*
_output_shapes
:T
add_2AddV2split:output:2MatMul_2:output:0*
T0*
_output_shapes
::
TanhTanh	add_2:z:0*
T0*
_output_shapes
:U
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype0Z
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:B
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:G
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:I
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:K

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�	
�
&__inference_signature_wrapper_32766323	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_32765142f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
"
_output_shapes
:

_user_specified_nameInput
�
�
E__inference_dense_2_layer_call_and_return_conditional_losses_32767608

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_model_layer_call_and_return_conditional_losses_32766190

inputs 
dense_32766164:
dense_32766166:
gru_32766169:
gru_32766171:
gru_32766173:
gru_32766175:"
dense_1_32766178:
dense_1_32766180:"
dense_2_32766183:
dense_2_32766185:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�gru/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32766164dense_32766166*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_32765643�
gru/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0gru_32766169gru_32766171gru_32766173gru_32766175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32766113�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_1_32766178dense_1_32766180*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32766183dense_2_32766185*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871�
flatten/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_32765883f
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_model_layer_call_and_return_conditional_losses_32765886

inputs 
dense_32765644:
dense_32765646:
gru_32765812:
gru_32765814:
gru_32765816:
gru_32765818:"
dense_1_32765844:
dense_1_32765846:"
dense_2_32765872:
dense_2_32765874:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�gru/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32765644dense_32765646*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_32765643�
gru/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0gru_32765812gru_32765814gru_32765816gru_32765818*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32765811�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_1_32765844dense_1_32765846*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32765872dense_2_32765874*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871�
flatten/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_32765883f
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_model_layer_call_and_return_conditional_losses_32766296	
input 
dense_32766270:
dense_32766272:
gru_32766275:
gru_32766277:
gru_32766279:
gru_32766281:"
dense_1_32766284:
dense_1_32766286:"
dense_2_32766289:
dense_2_32766291:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�gru/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputdense_32766270dense_32766272*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_32765643�
gru/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0gru_32766275gru_32766277gru_32766279gru_32766281*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32766113�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_1_32766284dense_1_32766286*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32766289dense_2_32766291*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871�
flatten/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_32765883f
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:I E
"
_output_shapes
:

_user_specified_nameInput
�W
�
A__inference_gru_layer_call_and_return_conditional_losses_32766113

inputs9
'gru_cell_matmul_readvariableop_resource:6
(gru_cell_biasadd_readvariableop_resource:2
 gru_cell_readvariableop_resource:;
)gru_cell_matmul_1_readvariableop_resource:
identity��AssignVariableOp�ReadVariableOp�gru_cell/BiasAdd/ReadVariableOp�gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�gru_cell/ReadVariableOp_1�gru_cell/mul/ReadVariableOp�gru_cell/mul_1/ReadVariableOp�whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          d
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:Z
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
gru_cell/BiasAdd/ReadVariableOpReadVariableOp(gru_cell_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0'gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitx
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0m
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0gru_cell/strided_slice:output:0*
T0*
_output_shapes

:c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/MatMul_1:product:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:V
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:t
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:Z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru_cell/mul/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0y
gru_cell/mulMulgru_cell/Sigmoid_1:y:0#gru_cell/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:z
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:*
dtype0o
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_masky
gru_cell/MatMul_2MatMulgru_cell/mul:z:0!gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:v
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:R
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru_cell/mul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0{
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:c
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:h
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : x
ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'gru_cell_matmul_readvariableop_resource(gru_cell_biasadd_readvariableop_resource gru_cell_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_32766017*
condR
while_cond_32766016*/
output_shapes
: : : : :: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOp)gru_cell_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Y
IdentityIdentitytranspose_1:y:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp^AssignVariableOp^ReadVariableOp ^gru_cell/BiasAdd/ReadVariableOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/mul/ReadVariableOp^gru_cell/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2B
gru_cell/BiasAdd/ReadVariableOpgru_cell/BiasAdd/ReadVariableOp2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_12:
gru_cell/mul/ReadVariableOpgru_cell/mul/ReadVariableOp2>
gru_cell/mul_1/ReadVariableOpgru_cell/mul_1/ReadVariableOp2
whilewhile:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�D
�
while_body_32766017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_gru_cell_matmul_readvariableop_resource_0:>
0while_gru_cell_biasadd_readvariableop_resource_0::
(while_gru_cell_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_gru_cell_matmul_readvariableop_resource:<
.while_gru_cell_biasadd_readvariableop_resource:8
&while_gru_cell_readvariableop_resource:��%while/gru_cell/BiasAdd/ReadVariableOp�$while/gru_cell/MatMul/ReadVariableOp�while/gru_cell/ReadVariableOp�while/gru_cell/ReadVariableOp_1�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp0while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0-while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0s
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2%while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/MatMul_1:product:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes

:b
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes

:�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes

:f
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes

:u
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes

:�
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0u
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2!while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:^
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes

:u
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:u
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:z
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: l
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*
_output_shapes

:�

while/NoOpNoOp&^while/gru_cell/BiasAdd/ReadVariableOp%^while/gru_cell/MatMul/ReadVariableOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "b
.while_gru_cell_biasadd_readvariableop_resource0while_gru_cell_biasadd_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2N
%while/gru_cell/BiasAdd/ReadVariableOp%while/gru_cell/BiasAdd/ReadVariableOp2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�	
�
(__inference_model_layer_call_fn_32766238	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_32766190f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
"
_output_shapes
:

_user_specified_nameInput
�
�
&__inference_gru_layer_call_fn_32766868
inputs_0
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32765603s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0
�(
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765492

inputs
states:0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�mul/ReadVariableOp�mul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0t
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0strided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:Y
mul/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0^
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:[
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0`
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�J
�
gru_while_body_32766458$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2!
gru_while_gru_strided_slice_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0E
3gru_while_gru_cell_matmul_readvariableop_resource_0:B
4gru_while_gru_cell_biasadd_readvariableop_resource_0:>
,gru_while_gru_cell_readvariableop_resource_0:
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_gru_strided_slice]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorC
1gru_while_gru_cell_matmul_readvariableop_resource:@
2gru_while_gru_cell_biasadd_readvariableop_resource:<
*gru_while_gru_cell_readvariableop_resource:��)gru/while/gru_cell/BiasAdd/ReadVariableOp�(gru/while/gru_cell/MatMul/ReadVariableOp�!gru/while/gru_cell/ReadVariableOp�#gru/while/gru_cell/ReadVariableOp_1�
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0�
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)gru/while/gru_cell/BiasAdd/ReadVariableOpReadVariableOp4gru_while_gru_cell_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:01gru/while/gru_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:m
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_split�
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 gru/while/gru_cell/strided_sliceStridedSlice)gru/while/gru_cell/ReadVariableOp:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_2)gru/while/gru_cell/strided_slice:output:0*
T0*
_output_shapes

:m
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����o
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/MatMul_1:product:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_split�
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*
_output_shapes

:j
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*
_output_shapes

:n
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0gru_while_placeholder_2*
T0*
_output_shapes

:�
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask�
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2%gru/while/gru_cell/MatMul_2:product:0*
T0*
_output_shapes

:f
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*
_output_shapes

:]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:�
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*
_output_shapes

:�
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*
_output_shapes

:�
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: �
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: x
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*
_output_shapes

:�
gru/while/NoOpNoOp*^gru/while/gru_cell/BiasAdd/ReadVariableOp)^gru/while/gru_cell/MatMul/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "j
2gru_while_gru_cell_biasadd_readvariableop_resource4gru_while_gru_cell_biasadd_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"<
gru_while_gru_strided_slicegru_while_gru_strided_slice_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"�
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : :: : : : : 2V
)gru/while/gru_cell/BiasAdd/ReadVariableOp)gru/while/gru_cell/BiasAdd/ReadVariableOp2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_1: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_layer_call_and_return_conditional_losses_32766842

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      o
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*"
_output_shapes
:P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?l
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*"
_output_shapes
:N
Gelu/ErfErfGelu/truediv:z:0*
T0*"
_output_shapes
:O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*"
_output_shapes
:Z

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*"
_output_shapes
:X
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*"
_output_shapes
:z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
while_cond_32765538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32765538___redundant_placeholder06
2while_while_cond_32765538___redundant_placeholder16
2while_while_cond_32765538___redundant_placeholder26
2while_while_cond_32765538___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�
�
while_cond_32767449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice6
2while_while_cond_32767449___redundant_placeholder06
2while_while_cond_32767449___redundant_placeholder16
2while_while_cond_32767449___redundant_placeholder26
2while_while_cond_32767449___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : :: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
�(
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32765213

inputs
states:0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:)
readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�mul/ReadVariableOp�mul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:::*
	num_splitf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0t
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0strided_slice:output:0*
T0*
_output_shapes

:Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*0
_output_shapes
::: *
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:D
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:Y
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:H
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:Y
mul/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0^
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask^
MatMul_2MatMulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes

:[
add_2AddV2split:output:2MatMul_2:product:0*
T0*
_output_shapes

:@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:[
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:*
dtype0`
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:H
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:M
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:O
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes

:Q

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�+
�
$__inference__traced_restore_32767962
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:3
!assignvariableop_2_dense_1_kernel:-
assignvariableop_3_dense_1_bias:3
!assignvariableop_4_dense_2_kernel:-
assignvariableop_5_dense_2_bias:8
&assignvariableop_6_gru_gru_cell_kernel:B
0assignvariableop_7_gru_gru_cell_recurrent_kernel:2
$assignvariableop_8_gru_gru_cell_bias:1
assignvariableop_9_gru_variable:
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_gru_gru_cell_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_gru_gru_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_gru_gru_cell_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_gru_variableIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
C__inference_model_layer_call_and_return_conditional_losses_32766267	
input 
dense_32766241:
dense_32766243:
gru_32766246:
gru_32766248:
gru_32766250:
gru_32766252:"
dense_1_32766255:
dense_1_32766257:"
dense_2_32766260:
dense_2_32766262:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�gru/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputdense_32766241dense_32766243*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_32765643�
gru/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0gru_32766246gru_32766248gru_32766250gru_32766252*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32765811�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_1_32766255dense_1_32766257*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_32765843�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32766260dense_2_32766262*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_32765871�
flatten/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_32765883f
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:I E
"
_output_shapes
:

_user_specified_nameInput
�
�
&__inference_gru_layer_call_fn_32766855
inputs_0
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_32765339s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0
�	
�
(__inference_model_layer_call_fn_32765909	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_32765886f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
"
_output_shapes
:

_user_specified_nameInput"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
2
Input)
serving_default_Input:02
flatten'
StatefulPartitionedCall:0tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
_
0
1
62
73
84
&5
'6
.7
/8"
trackable_list_wrapper
_
0
1
62
73
84
&5
'6
.7
/8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_0
?trace_1
@trace_2
Atrace_32�
(__inference_model_layer_call_fn_32765909
(__inference_model_layer_call_fn_32766348
(__inference_model_layer_call_fn_32766373
(__inference_model_layer_call_fn_32766238�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z>trace_0z?trace_1z@trace_2zAtrace_3
�
Btrace_0
Ctrace_1
Dtrace_2
Etrace_32�
C__inference_model_layer_call_and_return_conditional_losses_32766592
C__inference_model_layer_call_and_return_conditional_losses_32766811
C__inference_model_layer_call_and_return_conditional_losses_32766267
C__inference_model_layer_call_and_return_conditional_losses_32766296�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zBtrace_0zCtrace_1zDtrace_2zEtrace_3
�B�
#__inference__wrapped_model_32765142Input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Fserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
(__inference_dense_layer_call_fn_32766820�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
�
Mtrace_02�
C__inference_dense_layer_call_and_return_conditional_losses_32766842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
:2dense/kernel
:2
dense/bias
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Nstates
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32�
&__inference_gru_layer_call_fn_32766855
&__inference_gru_layer_call_fn_32766868
&__inference_gru_layer_call_fn_32766881
&__inference_gru_layer_call_fn_32766894�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
�
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32�
A__inference_gru_layer_call_and_return_conditional_losses_32767057
A__inference_gru_layer_call_and_return_conditional_losses_32767220
A__inference_gru_layer_call_and_return_conditional_losses_32767383
A__inference_gru_layer_call_and_return_conditional_losses_32767546�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
"
_generic_user_object
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator

6kernel
7recurrent_kernel
8bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
*__inference_dense_1_layer_call_fn_32767555�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
�
itrace_02�
E__inference_dense_1_layer_call_and_return_conditional_losses_32767577�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
 :2dense_1/kernel
:2dense_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
*__inference_dense_2_layer_call_fn_32767586�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
�
ptrace_02�
E__inference_dense_2_layer_call_and_return_conditional_losses_32767608�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
 :2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
*__inference_flatten_layer_call_fn_32767613�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
�
wtrace_02�
E__inference_flatten_layer_call_and_return_conditional_losses_32767619�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
%:#2gru/gru_cell/kernel
/:-2gru/gru_cell/recurrent_kernel
:2gru/gru_cell/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_layer_call_fn_32765909Input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_layer_call_fn_32766348inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_layer_call_fn_32766373inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_layer_call_fn_32766238Input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_32766592inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_32766811inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_32766267Input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_32766296Input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
&__inference_signature_wrapper_32766323Input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_layer_call_fn_32766820inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_layer_call_and_return_conditional_losses_32766842inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_gru_layer_call_fn_32766855inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
&__inference_gru_layer_call_fn_32766868inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
&__inference_gru_layer_call_fn_32766881inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
&__inference_gru_layer_call_fn_32766894inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
A__inference_gru_layer_call_and_return_conditional_losses_32767057inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
A__inference_gru_layer_call_and_return_conditional_losses_32767220inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
A__inference_gru_layer_call_and_return_conditional_losses_32767383inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
A__inference_gru_layer_call_and_return_conditional_losses_32767546inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
~trace_0
trace_1
�trace_2
�trace_32�
+__inference_gru_cell_layer_call_fn_32767633
+__inference_gru_cell_layer_call_fn_32767647
+__inference_gru_cell_layer_call_fn_32767662
+__inference_gru_cell_layer_call_fn_32767677�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z~trace_0ztrace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767727
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767773
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767819
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767869�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_1_layer_call_fn_32767555inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_1_layer_call_and_return_conditional_losses_32767577inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_2_layer_call_fn_32767586inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_2_layer_call_and_return_conditional_losses_32767608inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_layer_call_fn_32767613inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_layer_call_and_return_conditional_losses_32767619inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:2gru/Variable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_gru_cell_layer_call_fn_32767633inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
+__inference_gru_cell_layer_call_fn_32767647inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
+__inference_gru_cell_layer_call_fn_32767662inputsstates"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
+__inference_gru_cell_layer_call_fn_32767677inputsstates"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767727inputsstates"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767773inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767819inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767869inputsstates"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
#__inference__wrapped_model_32765142a
687x&'./)�&
�
�
Input
� "(�%
#
flatten�
flatten�
E__inference_dense_1_layer_call_and_return_conditional_losses_32767577R&'*�'
 �
�
inputs
� " �
�
0
� s
*__inference_dense_1_layer_call_fn_32767555E&'*�'
 �
�
inputs
� "��
E__inference_dense_2_layer_call_and_return_conditional_losses_32767608R./*�'
 �
�
inputs
� " �
�
0
� s
*__inference_dense_2_layer_call_fn_32767586E./*�'
 �
�
inputs
� "��
C__inference_dense_layer_call_and_return_conditional_losses_32766842R*�'
 �
�
inputs
� " �
�
0
� q
(__inference_dense_layer_call_fn_32766820E*�'
 �
�
inputs
� "��
E__inference_flatten_layer_call_and_return_conditional_losses_32767619J*�'
 �
�
inputs
� "�
�
0
� k
*__inference_flatten_layer_call_fn_32767613=*�'
 �
�
inputs
� "��
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767727�687a�^
W�T
�
inputs
5�2
0�-	�
�
�
pVariableSpec 
p 
� "4�1
*�'
�
0/0
�
�
0/1/0
� �
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767773�687J�G
@�=
�
inputs
�
�
states/0
p 
� "@�=
6�3
�
0/0
�
�
0/1/0
� �
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767819�687J�G
@�=
�
inputs
�
�
states/0
p
� "@�=
6�3
�
0/0
�
�
0/1/0
� �
F__inference_gru_cell_layer_call_and_return_conditional_losses_32767869�687a�^
W�T
�
inputs
5�2
0�-	�
�
�
pVariableSpec 
p
� "4�1
*�'
�
0/0
�
�
0/1/0
� �
+__inference_gru_cell_layer_call_fn_32767633�687J�G
@�=
�
inputs
�
�
states/0
p 
� "2�/
�
0
�
�
1/0�
+__inference_gru_cell_layer_call_fn_32767647�687J�G
@�=
�
inputs
�
�
states/0
p
� "2�/
�
0
�
�
1/0�
+__inference_gru_cell_layer_call_fn_32767662�687a�^
W�T
�
inputs
5�2
0�-	�
�
�
pVariableSpec 
p 
� "2�/
�
0
�
�
1/0�
+__inference_gru_cell_layer_call_fn_32767677�687a�^
W�T
�
inputs
5�2
0�-	�
�
�
pVariableSpec 
p
� "2�/
�
0
�
�
1/0�
A__inference_gru_layer_call_and_return_conditional_losses_32767057y687xF�C
<�9
+�(
&�#
inputs/0���������

 
p 

 
� ")�&
�
0���������
� �
A__inference_gru_layer_call_and_return_conditional_losses_32767220y687xF�C
<�9
+�(
&�#
inputs/0���������

 
p

 
� ")�&
�
0���������
� �
A__inference_gru_layer_call_and_return_conditional_losses_32767383`687x6�3
,�)
�
inputs

 
p 

 
� " �
�
0
� �
A__inference_gru_layer_call_and_return_conditional_losses_32767546`687x6�3
,�)
�
inputs

 
p

 
� " �
�
0
� �
&__inference_gru_layer_call_fn_32766855lx687F�C
<�9
+�(
&�#
inputs/0���������

 
p 

 
� "�����������
&__inference_gru_layer_call_fn_32766868lx687F�C
<�9
+�(
&�#
inputs/0���������

 
p

 
� "����������}
&__inference_gru_layer_call_fn_32766881S687x6�3
,�)
�
inputs

 
p 

 
� "�}
&__inference_gru_layer_call_fn_32766894S687x6�3
,�)
�
inputs

 
p

 
� "��
C__inference_model_layer_call_and_return_conditional_losses_32766267]
687x&'./1�.
'�$
�
Input
p 

 
� "�
�
0
� �
C__inference_model_layer_call_and_return_conditional_losses_32766296]
687x&'./1�.
'�$
�
Input
p

 
� "�
�
0
� �
C__inference_model_layer_call_and_return_conditional_losses_32766592^
687x&'./2�/
(�%
�
inputs
p 

 
� "�
�
0
� �
C__inference_model_layer_call_and_return_conditional_losses_32766811^
687x&'./2�/
(�%
�
inputs
p

 
� "�
�
0
� |
(__inference_model_layer_call_fn_32765909P
687x&'./1�.
'�$
�
Input
p 

 
� "�|
(__inference_model_layer_call_fn_32766238P
687x&'./1�.
'�$
�
Input
p

 
� "�}
(__inference_model_layer_call_fn_32766348Q
687x&'./2�/
(�%
�
inputs
p 

 
� "�}
(__inference_model_layer_call_fn_32766373Q
687x&'./2�/
(�%
�
inputs
p

 
� "��
&__inference_signature_wrapper_32766323j
687x&'./2�/
� 
(�%
#
Input�
Input"(�%
#
flatten�
flatten