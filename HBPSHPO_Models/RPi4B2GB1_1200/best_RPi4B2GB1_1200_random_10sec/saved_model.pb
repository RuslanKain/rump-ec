??!
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
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
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28׌
v
training/SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *"
shared_nametraining/SGD/iter
o
%training/SGD/iter/Read/ReadVariableOpReadVariableOptraining/SGD/iter*
_output_shapes
: *
dtype0	
x
training/SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nametraining/SGD/decay
q
&training/SGD/decay/Read/ReadVariableOpReadVariableOptraining/SGD/decay*
_output_shapes
: *
dtype0
?
training/SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nametraining/SGD/learning_rate
?
.training/SGD/learning_rate/Read/ReadVariableOpReadVariableOptraining/SGD/learning_rate*
_output_shapes
: *
dtype0
~
training/SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining/SGD/momentum
w
)training/SGD/momentum/Read/ReadVariableOpReadVariableOptraining/SGD/momentum*
_output_shapes
: *
dtype0
?
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namelstm/lstm_cell/kernel
?
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	?*
dtype0
?
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*0
shared_name!lstm/lstm_cell/recurrent_kernel
?
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:?*
dtype0
?
time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nametime_distributed/kernel
?
+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel* 
_output_shapes
:
??*
dtype0
?
time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nametime_distributed/bias
|
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes	
:?*
dtype0
?
time_distributed_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?p**
shared_nametime_distributed_2/kernel
?
-time_distributed_2/kernel/Read/ReadVariableOpReadVariableOptime_distributed_2/kernel*
_output_shapes
:	?p*
dtype0
?
time_distributed_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:p*(
shared_nametime_distributed_2/bias

+time_distributed_2/bias/Read/ReadVariableOpReadVariableOptime_distributed_2/bias*
_output_shapes
:p*
dtype0
?
time_distributed_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:p8**
shared_nametime_distributed_4/kernel
?
-time_distributed_4/kernel/Read/ReadVariableOpReadVariableOptime_distributed_4/kernel*
_output_shapes

:p8*
dtype0
?
time_distributed_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*(
shared_nametime_distributed_4/bias

+time_distributed_4/bias/Read/ReadVariableOpReadVariableOptime_distributed_4/bias*
_output_shapes
:8*
dtype0
?
time_distributed_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8**
shared_nametime_distributed_6/kernel
?
-time_distributed_6/kernel/Read/ReadVariableOpReadVariableOptime_distributed_6/kernel*
_output_shapes

:8*
dtype0
?
time_distributed_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_6/bias

+time_distributed_6/bias/Read/ReadVariableOpReadVariableOptime_distributed_6/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
]
	layer
	variables
trainable_variables
regularization_losses
	keras_api
]
	layer
	variables
trainable_variables
regularization_losses
	keras_api
]
	 layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
]
	%layer
&	variables
'trainable_variables
(regularization_losses
)	keras_api
]
	*layer
+	variables
,trainable_variables
-regularization_losses
.	keras_api
]
	/layer
0	variables
1trainable_variables
2regularization_losses
3	keras_api
]
	4layer
5	variables
6trainable_variables
7regularization_losses
8	keras_api
R
9	variables
:trainable_variables
;regularization_losses
<	keras_api
6
=iter
	>decay
?learning_rate
@momentum
N
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
N
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
 
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
 
?
Q
state_size

Akernel
Brecurrent_kernel
Cbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
 

A0
B1
C2

A0
B1
C2
 
?

Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
h

Dkernel
Ebias
\	variables
]trainable_variables
^regularization_losses
_	keras_api

D0
E1

D0
E1
 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
R
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
 
 
 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
h

Fkernel
Gbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api

F0
G1

F0
G1
 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
!	variables
"trainable_variables
#regularization_losses
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
 
 
 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
l

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

H0
I1

H0
I1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
l

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

J0
K1

J0
K1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
PN
VARIABLE_VALUEtraining/SGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEtraining/SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEtraining/SGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtraining/SGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEtime_distributed/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_2/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed_2/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_4/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed_4/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_6/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtime_distributed_6/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

?0
?1
 
 
 

A0
B1
C2

A0
B1
C2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
 
 

0
 
 
 

D0
E1

D0
E1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
 

0
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
 

0
 
 
 

F0
G1

F0
G1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
 

 0
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
 

%0
 
 
 

H0
I1

H0
I1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

*0
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

/0
 
 
 

J0
K1

J0
K1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

40
 
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_lstm_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastime_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/biastime_distributed_4/kerneltime_distributed_4/biastime_distributed_6/kerneltime_distributed_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_163667
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%training/SGD/iter/Read/ReadVariableOp&training/SGD/decay/Read/ReadVariableOp.training/SGD/learning_rate/Read/ReadVariableOp)training/SGD/momentum/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_2/kernel/Read/ReadVariableOp+time_distributed_2/bias/Read/ReadVariableOp-time_distributed_4/kernel/Read/ReadVariableOp+time_distributed_4/bias/Read/ReadVariableOp-time_distributed_6/kernel/Read/ReadVariableOp+time_distributed_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_165768
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining/SGD/itertraining/SGD/decaytraining/SGD/learning_ratetraining/SGD/momentumlstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastime_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/biastime_distributed_4/kerneltime_distributed_4/biastime_distributed_6/kerneltime_distributed_6/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_165835ג
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165118

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????pb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????p"
identityIdentity:output:0**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163083

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????8"
identityIdentity:output:0**
_input_shapes
:?????????8:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?J
?
@__inference_lstm_layer_call_and_return_conditional_losses_164476
inputs_0H
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164392*
condR
while_cond_164391*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?-
?
__inference__traced_save_165768
file_prefix0
,savev2_training_sgd_iter_read_readvariableop	1
-savev2_training_sgd_decay_read_readvariableop9
5savev2_training_sgd_learning_rate_read_readvariableop4
0savev2_training_sgd_momentum_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_2_kernel_read_readvariableop6
2savev2_time_distributed_2_bias_read_readvariableop8
4savev2_time_distributed_4_kernel_read_readvariableop6
2savev2_time_distributed_4_bias_read_readvariableop8
4savev2_time_distributed_6_kernel_read_readvariableop6
2savev2_time_distributed_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_training_sgd_iter_read_readvariableop-savev2_training_sgd_decay_read_readvariableop5savev2_training_sgd_learning_rate_read_readvariableop0savev2_training_sgd_momentum_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_2_kernel_read_readvariableop2savev2_time_distributed_2_bias_read_readvariableop4savev2_time_distributed_4_kernel_read_readvariableop2savev2_time_distributed_4_bias_read_readvariableop4savev2_time_distributed_6_kernel_read_readvariableop2savev2_time_distributed_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapesx
v: : : : : :	?:
??:?:
??:?:	?p:p:p8:8:8:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:%
!

_output_shapes
:	?p: 

_output_shapes
:p:$ 

_output_shapes

:p8: 

_output_shapes
:8:$ 

_output_shapes

:8: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_time_distributed_2_layer_call_fn_164962

inputs,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_162528|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_165559

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_162407a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
lstm_while_cond_163757
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_13
/lstm_while_cond_163757___redundant_placeholder03
/lstm_while_cond_163757___redundant_placeholder13
/lstm_while_cond_163757___redundant_placeholder23
/lstm_while_cond_163757___redundant_placeholder3
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_162437

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_4_layer_call_fn_165148

inputs+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_162673|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_162603

inputs
identity??!dropout_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_162582\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape*dropout_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????pj
NoOpNoOp"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
?
while_cond_164391
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_164391___redundant_placeholder0.
*while_cond_164391___redundant_placeholder1.
*while_cond_164391___redundant_placeholder2.
*while_cond_164391___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?=
?
F__inference_sequential_layer_call_and_return_conditional_losses_163611

lstm_input-
lstm_lstm_lstm_cell_kernel:	?8
$lstm_lstm_lstm_cell_recurrent_kernel:
??'
lstm_lstm_lstm_cell_bias:	?<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?p8
*time_distributed_2_time_distributed_2_bias:p>
,time_distributed_4_time_distributed_4_kernel:p88
*time_distributed_4_time_distributed_4_bias:8>
,time_distributed_6_time_distributed_6_kernel:88
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_162986?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163005o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshape%lstm/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163019q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0,time_distributed_2_time_distributed_2_kernel*time_distributed_2_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163037q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape+time_distributed_1/PartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163051q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163069q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape+time_distributed_3/PartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163083q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163100q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?
?
while_cond_162901
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_162901___redundant_placeholder0.
*while_cond_162901___redundant_placeholder1.
*while_cond_162901___redundant_placeholder2.
*while_cond_162901___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
1__inference_time_distributed_layer_call_fn_164790

inputs+
time_distributed_kernel:
??$
time_distributed_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_kerneltime_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163323t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164900

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????a
dropout/IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_4_layer_call_fn_165155

inputs+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163069s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_162407

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_163323

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162191

inputs

states
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?
!__inference__wrapped_model_161990

lstm_inputX
Esequential_lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?e
Qsequential_lstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??S
Dsequential_lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?c
Osequential_time_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??]
Nsequential_time_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?h
Usequential_time_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pb
Tsequential_time_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:pg
Usequential_time_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:p8b
Tsequential_time_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:8g
Usequential_time_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:8b
Tsequential_time_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp?/sequential/lstm/lstm_cell/MatMul/ReadVariableOp?1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp?sequential/lstm/while?8sequential/time_distributed/dense/BiasAdd/ReadVariableOp?7sequential/time_distributed/dense/MatMul/ReadVariableOp?<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp?;sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp?<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp?;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp?<sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp?;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOpO
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????c
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????s
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose	Transpose
lstm_input'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????d
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:o
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???o
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
/sequential/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOpEsequential_lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:07sequential/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpQsequential_lstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
"sequential/lstm/lstm_cell/MatMul_1MatMulsequential/lstm/zeros:output:09sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/MatMul:product:0,sequential/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDsequential_lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
!sequential/lstm/lstm_cell/BiasAddBiasAdd!sequential/lstm/lstm_cell/add:z:08sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:0*sequential/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
!sequential/lstm/lstm_cell/SigmoidSigmoid(sequential/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid(sequential/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mulMul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/ReluRelu(sequential/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mul_1Mul%sequential/lstm/lstm_cell/Sigmoid:y:0,sequential/lstm/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/add_1AddV2!sequential/lstm/lstm_cell/mul:z:0#sequential/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid(sequential/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
 sequential/lstm/lstm_cell/Relu_1Relu#sequential/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mul_2Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0.sequential/lstm/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???V
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Esequential_lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelQsequential_lstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelDsequential_lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!sequential_lstm_while_body_161830*-
cond%R#
!sequential_lstm_while_cond_161829*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0x
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????k
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    z
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
#sequential/time_distributed/ReshapeReshapesequential/lstm/transpose_1:y:02sequential/time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
7sequential/time_distributed/dense/MatMul/ReadVariableOpReadVariableOpOsequential_time_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
(sequential/time_distributed/dense/MatMulMatMul,sequential/time_distributed/Reshape:output:0?sequential/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
8sequential/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpNsequential_time_distributed_dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
)sequential/time_distributed/dense/BiasAddBiasAdd2sequential/time_distributed/dense/MatMul:product:0@sequential/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&sequential/time_distributed/dense/ReluRelu2sequential/time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+sequential/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
%sequential/time_distributed/Reshape_1Reshape4sequential/time_distributed/dense/Relu:activations:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????|
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
%sequential/time_distributed/Reshape_2Reshapesequential/lstm/transpose_1:y:04sequential/time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????|
+sequential/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
%sequential/time_distributed_1/ReshapeReshape.sequential/time_distributed/Reshape_1:output:04sequential/time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
.sequential/time_distributed_1/dropout/IdentityIdentity.sequential/time_distributed_1/Reshape:output:0*
T0*(
_output_shapes
:???????????
-sequential/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
'sequential/time_distributed_1/Reshape_1Reshape7sequential/time_distributed_1/dropout/Identity:output:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????~
-sequential/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'sequential/time_distributed_1/Reshape_2Reshape.sequential/time_distributed/Reshape_1:output:06sequential/time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????|
+sequential/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
%sequential/time_distributed_2/ReshapeReshape0sequential/time_distributed_1/Reshape_1:output:04sequential/time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
;sequential/time_distributed_2/dense_1/MatMul/ReadVariableOpReadVariableOpUsequential_time_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
,sequential/time_distributed_2/dense_1/MatMulMatMul.sequential/time_distributed_2/Reshape:output:0Csequential/time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpTsequential_time_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
-sequential/time_distributed_2/dense_1/BiasAddBiasAdd6sequential/time_distributed_2/dense_1/MatMul:product:0Dsequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
*sequential/time_distributed_2/dense_1/ReluRelu6sequential/time_distributed_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p?
-sequential/time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
'sequential/time_distributed_2/Reshape_1Reshape8sequential/time_distributed_2/dense_1/Relu:activations:06sequential/time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p~
-sequential/time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'sequential/time_distributed_2/Reshape_2Reshape0sequential/time_distributed_1/Reshape_1:output:06sequential/time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????|
+sequential/time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
%sequential/time_distributed_3/ReshapeReshape0sequential/time_distributed_2/Reshape_1:output:04sequential/time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
0sequential/time_distributed_3/dropout_1/IdentityIdentity.sequential/time_distributed_3/Reshape:output:0*
T0*'
_output_shapes
:?????????p?
-sequential/time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
'sequential/time_distributed_3/Reshape_1Reshape9sequential/time_distributed_3/dropout_1/Identity:output:06sequential/time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p~
-sequential/time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
'sequential/time_distributed_3/Reshape_2Reshape0sequential/time_distributed_2/Reshape_1:output:06sequential/time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????p|
+sequential/time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
%sequential/time_distributed_4/ReshapeReshape0sequential/time_distributed_3/Reshape_1:output:04sequential/time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpUsequential_time_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
,sequential/time_distributed_4/dense_2/MatMulMatMul.sequential/time_distributed_4/Reshape:output:0Csequential/time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpTsequential_time_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
-sequential/time_distributed_4/dense_2/BiasAddBiasAdd6sequential/time_distributed_4/dense_2/MatMul:product:0Dsequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
*sequential/time_distributed_4/dense_2/ReluRelu6sequential/time_distributed_4/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8?
-sequential/time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
'sequential/time_distributed_4/Reshape_1Reshape8sequential/time_distributed_4/dense_2/Relu:activations:06sequential/time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8~
-sequential/time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
'sequential/time_distributed_4/Reshape_2Reshape0sequential/time_distributed_3/Reshape_1:output:06sequential/time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????p|
+sequential/time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
%sequential/time_distributed_5/ReshapeReshape0sequential/time_distributed_4/Reshape_1:output:04sequential/time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
0sequential/time_distributed_5/dropout_2/IdentityIdentity.sequential/time_distributed_5/Reshape:output:0*
T0*'
_output_shapes
:?????????8?
-sequential/time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
'sequential/time_distributed_5/Reshape_1Reshape9sequential/time_distributed_5/dropout_2/Identity:output:06sequential/time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8~
-sequential/time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
'sequential/time_distributed_5/Reshape_2Reshape0sequential/time_distributed_4/Reshape_1:output:06sequential/time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8|
+sequential/time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
%sequential/time_distributed_6/ReshapeReshape0sequential/time_distributed_5/Reshape_1:output:04sequential/time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpUsequential_time_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
,sequential/time_distributed_6/dense_3/MatMulMatMul.sequential/time_distributed_6/Reshape:output:0Csequential/time_distributed_6/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
<sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOpReadVariableOpTsequential_time_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
-sequential/time_distributed_6/dense_3/BiasAddBiasAdd6sequential/time_distributed_6/dense_3/MatMul:product:0Dsequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential/time_distributed_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
'sequential/time_distributed_6/Reshape_1Reshape6sequential/time_distributed_6/dense_3/BiasAdd:output:06sequential/time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????~
-sequential/time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
'sequential/time_distributed_6/Reshape_2Reshape0sequential/time_distributed_5/Reshape_1:output:06sequential/time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8~
)sequential/cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ?
+sequential/cropping1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            ?
+sequential/cropping1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
#sequential/cropping1d/strided_sliceStridedSlice0sequential/time_distributed_6/Reshape_1:output:02sequential/cropping1d/strided_slice/stack:output:04sequential/cropping1d/strided_slice/stack_1:output:04sequential/cropping1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask
IdentityIdentity,sequential/cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp1^sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp0^sequential/lstm/lstm_cell/MatMul/ReadVariableOp2^sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp^sequential/lstm/while9^sequential/time_distributed/dense/BiasAdd/ReadVariableOp8^sequential/time_distributed/dense/MatMul/ReadVariableOp=^sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp<^sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp=^sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp<^sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp=^sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp<^sequential/time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2d
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp2b
/sequential/lstm/lstm_cell/MatMul/ReadVariableOp/sequential/lstm/lstm_cell/MatMul/ReadVariableOp2f
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while2t
8sequential/time_distributed/dense/BiasAdd/ReadVariableOp8sequential/time_distributed/dense/BiasAdd/ReadVariableOp2r
7sequential/time_distributed/dense/MatMul/ReadVariableOp7sequential/time_distributed/dense/MatMul/ReadVariableOp2|
<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp2z
;sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp;sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp2|
<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp2z
;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp2|
<sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp<sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp2z
;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOp;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOp:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164864

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_162341

inputsA
-matmul_readvariableop_time_distributed_kernel:
??;
,biasadd_readvariableop_time_distributed_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_162784

inputs3
!dense_3_time_distributed_6_kernel:8-
dense_3_time_distributed_6_bias:
identity??dense_3/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_3_time_distributed_6_kerneldense_3_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_162775\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????h
NoOpNoOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163213

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165221

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_5_layer_call_fn_165256

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163186s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????822
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_3_layer_call_fn_165060

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_162603|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_165626

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????pC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????po
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????pi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????pY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????p"
identityIdentity:output:0*&
_input_shapes
:?????????p:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163296

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??
dropout/dropout/MulMulReshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????U
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
	Reshape_1Reshapedropout/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_162350

inputs1
dense_time_distributed_kernel:
??*
dense_time_distributed_bias:	?
identity??dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_time_distributed_kerneldense_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_162341\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_2_layer_call_fn_164969

inputs,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163037s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_5_layer_call_fn_165246

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_162748|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????822
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_163005

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165536

inputs
states_0
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164834

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_162414

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dropout/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_162407\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape dropout/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_163396
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_163396___redundant_placeholder0.
*while_cond_163396___redundant_placeholder1.
*while_cond_163396___redundant_placeholder2.
*while_cond_163396___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
3__inference_time_distributed_6_layer_call_fn_165334

inputs+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_162817|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_165341

inputs+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163100s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_162383

inputs1
dense_time_distributed_kernel:
??*
dense_time_distributed_bias:	?
identity??dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_time_distributed_kerneldense_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_162341\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165020

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
*__inference_lstm_cell_layer_call_fn_165458

inputs
states_0
states_1(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162057p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165206

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
?
1__inference_time_distributed_layer_call_fn_164776

inputs+
time_distributed_kernel:
??$
time_distributed_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_kerneltime_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_162383}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_164174
inputs_0(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_162310}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_164677
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_164677___redundant_placeholder0.
*while_cond_164677___redundant_placeholder1.
*while_cond_164677___redundant_placeholder2.
*while_cond_164677___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165369

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_165649

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_162697`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????8"
identityIdentity:output:0*&
_input_shapes
:?????????8:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
while_cond_162067
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_162067___redundant_placeholder0.
*while_cond_162067___redundant_placeholder1.
*while_cond_162067___redundant_placeholder2.
*while_cond_162067___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_165599

inputsB
/matmul_readvariableop_time_distributed_2_kernel:	?p<
.biasadd_readvariableop_time_distributed_2_bias:p
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????pP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????pa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_165564

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_162437p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_4_layer_call_fn_165141

inputs+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_162640|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_165671

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????8C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????8"
identityIdentity:output:0*&
_input_shapes
:?????????8:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_163918

inputsM
:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?Z
Flstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??H
9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?X
Dtime_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??R
Ctime_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?]
Jtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pW
Itime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:p\
Jtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:p8W
Itime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:8\
Jtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:8W
Itime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??%lstm/lstm_cell/BiasAdd/ReadVariableOp?$lstm/lstm_cell/MatMul/ReadVariableOp?&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_2/dense_1/BiasAdd/ReadVariableOp?0time_distributed_2/dense_1/MatMul/ReadVariableOp?1time_distributed_4/dense_2/BiasAdd/ReadVariableOp?0time_distributed_4/dense_2/MatMul/ReadVariableOp?1time_distributed_6/dense_3/BiasAdd/ReadVariableOp?0time_distributed_6/dense_3/MatMul/ReadVariableOp@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splits
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????m
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????j
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_163758*"
condR
lstm_while_cond_163757*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshapelstm/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOpDtime_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpCtime_distributed_dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed/Reshape_1Reshape)time_distributed/dense/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????q
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/Reshape_2Reshapelstm/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
#time_distributed_1/dropout/IdentityIdentity#time_distributed_1/Reshape:output:0*
T0*(
_output_shapes
:??????????w
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed_1/Reshape_1Reshape,time_distributed_1/dropout/Identity:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????s
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
0time_distributed_2/dense_1/MatMul/ReadVariableOpReadVariableOpJtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
!time_distributed_2/dense_1/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
1time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
"time_distributed_2/dense_1/BiasAddBiasAdd+time_distributed_2/dense_1/MatMul:product:09time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
time_distributed_2/dense_1/ReluRelu+time_distributed_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
time_distributed_2/Reshape_1Reshape-time_distributed_2/dense_1/Relu:activations:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????ps
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
%time_distributed_3/dropout_1/IdentityIdentity#time_distributed_3/Reshape:output:0*
T0*'
_output_shapes
:?????????pw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
time_distributed_3/Reshape_1Reshape.time_distributed_3/dropout_1/Identity:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????ps
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????pq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
0time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpJtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
!time_distributed_4/dense_2/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
1time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
"time_distributed_4/dense_2/BiasAddBiasAdd+time_distributed_4/dense_2/MatMul:product:09time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
time_distributed_4/dense_2/ReluRelu+time_distributed_4/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
time_distributed_4/Reshape_1Reshape-time_distributed_4/dense_2/Relu:activations:0+time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/Reshape_2Reshape%time_distributed_3/Reshape_1:output:0+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????pq
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
%time_distributed_5/dropout_2/IdentityIdentity#time_distributed_5/Reshape:output:0*
T0*'
_output_shapes
:?????????8w
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
time_distributed_5/Reshape_1Reshape.time_distributed_5/dropout_2/Identity:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
0time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpJtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
!time_distributed_6/dense_3/MatMulMatMul#time_distributed_6/Reshape:output:08time_distributed_6/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1time_distributed_6/dense_3/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
"time_distributed_6/dense_3/BiasAddBiasAdd+time_distributed_6/dense_3/MatMul:product:09time_distributed_6/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
"time_distributed_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
time_distributed_6/Reshape_1Reshape+time_distributed_6/dense_3/BiasAdd:output:0+time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????s
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8s
cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            u
 cropping1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            u
 cropping1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
cropping1d/strided_sliceStridedSlice%time_distributed_6/Reshape_1:output:0'cropping1d/strided_slice/stack:output:0)cropping1d/strided_slice/stack_1:output:0)cropping1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_maskt
IdentityIdentity!cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_2/dense_1/BiasAdd/ReadVariableOp1^time_distributed_2/dense_1/MatMul/ReadVariableOp2^time_distributed_4/dense_2/BiasAdd/ReadVariableOp1^time_distributed_4/dense_2/MatMul/ReadVariableOp2^time_distributed_6/dense_3/BiasAdd/ReadVariableOp1^time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2f
1time_distributed_2/dense_1/BiasAdd/ReadVariableOp1time_distributed_2/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_2/dense_1/MatMul/ReadVariableOp0time_distributed_2/dense_1/MatMul/ReadVariableOp2f
1time_distributed_4/dense_2/BiasAdd/ReadVariableOp1time_distributed_4/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_4/dense_2/MatMul/ReadVariableOp0time_distributed_4/dense_2/MatMul/ReadVariableOp2f
1time_distributed_6/dense_3/BiasAdd/ReadVariableOp1time_distributed_6/dense_3/BiasAdd/ReadVariableOp2d
0time_distributed_6/dense_3/MatMul/ReadVariableOp0time_distributed_6/dense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165504

inputs
states_0
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
G
+__inference_cropping1d_layer_call_fn_165428

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
"__inference__traced_restore_165835
file_prefix,
"assignvariableop_training_sgd_iter:	 /
%assignvariableop_1_training_sgd_decay: 7
-assignvariableop_2_training_sgd_learning_rate: 2
(assignvariableop_3_training_sgd_momentum: ;
(assignvariableop_4_lstm_lstm_cell_kernel:	?F
2assignvariableop_5_lstm_lstm_cell_recurrent_kernel:
??5
&assignvariableop_6_lstm_lstm_cell_bias:	?>
*assignvariableop_7_time_distributed_kernel:
??7
(assignvariableop_8_time_distributed_bias:	??
,assignvariableop_9_time_distributed_2_kernel:	?p9
+assignvariableop_10_time_distributed_2_bias:p?
-assignvariableop_11_time_distributed_4_kernel:p89
+assignvariableop_12_time_distributed_4_bias:8?
-assignvariableop_13_time_distributed_6_kernel:89
+assignvariableop_14_time_distributed_6_bias:#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: 
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOp"assignvariableop_training_sgd_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp%assignvariableop_1_training_sgd_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_training_sgd_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_training_sgd_momentumIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_lstm_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp2assignvariableop_5_lstm_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_lstm_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_time_distributed_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_time_distributed_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_time_distributed_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_time_distributed_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_time_distributed_4_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_time_distributed_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_time_distributed_6_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_time_distributed_6_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
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
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_165569

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_1_layer_call_fn_164874

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_162458}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*4
_input_shapes#
!:???????????????????22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
1__inference_time_distributed_layer_call_fn_164769

inputs+
time_distributed_kernel:
??$
time_distributed_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_kerneltime_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_162350}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
1__inference_time_distributed_layer_call_fn_164783

inputs+
time_distributed_kernel:
??$
time_distributed_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_kerneltime_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163005t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164948

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??
dropout/dropout/MulMulReshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????U
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
	Reshape_1Reshapedropout/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_165543

inputs+
time_distributed_kernel:
??$
time_distributed_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_kerneltime_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_162341p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_162631

inputsA
/matmul_readvariableop_time_distributed_4_kernel:p8<
.biasadd_readvariableop_time_distributed_4_bias:8
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????8a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????8w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163100

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_2_layer_call_fn_164976

inputs,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163268s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_body_162068
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!lstm_cell_lstm_lstm_cell_kernel_0:	??
+lstm_cell_lstm_lstm_cell_recurrent_kernel_0:
??.
lstm_cell_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
lstm_cell_lstm_lstm_cell_kernel:	?=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??,
lstm_cell_lstm_lstm_cell_bias:	???!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3!lstm_cell_lstm_lstm_cell_kernel_0+lstm_cell_lstm_lstm_cell_recurrent_kernel_0lstm_cell_lstm_lstm_cell_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162057?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: |

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????|

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????j
NoOpNoOp"^lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"@
lstm_cell_lstm_lstm_cell_biaslstm_cell_lstm_lstm_cell_bias_0"D
lstm_cell_lstm_lstm_cell_kernel!lstm_cell_lstm_lstm_cell_kernel_0"X
)lstm_cell_lstm_lstm_cell_recurrent_kernel+lstm_cell_lstm_lstm_cell_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
c
*__inference_dropout_1_layer_call_fn_165609

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_162582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*&
_input_shapes
:?????????p22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_163683

inputs(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_163116s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_165614

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????p[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????p"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????p:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_165654

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_162727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*&
_input_shapes
:?????????822
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_3_layer_call_fn_165065

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163051d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????p"
identityIdentity:output:0**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
while_cond_162243
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_162243___redundant_placeholder0.
*while_cond_162243___redundant_placeholder1.
*while_cond_162243___redundant_placeholder2.
*while_cond_162243___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_body_162244
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!lstm_cell_lstm_lstm_cell_kernel_0:	??
+lstm_cell_lstm_lstm_cell_recurrent_kernel_0:
??.
lstm_cell_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
lstm_cell_lstm_lstm_cell_kernel:	?=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??,
lstm_cell_lstm_lstm_cell_bias:	???!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3!lstm_cell_lstm_lstm_cell_kernel_0+lstm_cell_lstm_lstm_cell_recurrent_kernel_0lstm_cell_lstm_lstm_cell_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162191?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: |

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????|

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????j
NoOpNoOp"^lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"@
lstm_cell_lstm_lstm_cell_biaslstm_cell_lstm_lstm_cell_bias_0"D
lstm_cell_lstm_lstm_cell_kernel!lstm_cell_lstm_lstm_cell_kernel_0"X
)lstm_cell_lstm_lstm_cell_recurrent_kernel+lstm_cell_lstm_lstm_cell_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165304

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????8"
identityIdentity:output:0**
_input_shapes
:?????????8:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_162552

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????p[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????p"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????p:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_165327

inputs+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_162784|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?1
?
while_body_164535
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_cropping1d_layer_call_fn_165423

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_162834v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_2_layer_call_fn_164955

inputs,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_162495|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165404

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?3
?
!sequential_lstm_while_body_161830&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!sequential_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5#
sequential_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :e
add_1AddV2"sequential_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: h

Identity_1Identity(sequential_lstm_while_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"D
sequential_lstm_strided_slice_1!sequential_lstm_strided_slice_1_0"?
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164932

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????a
dropout/IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
while_body_162902
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163069

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_163130

lstm_input(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_163116s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_162495

inputs4
!dense_1_time_distributed_2_kernel:	?p-
dense_1_time_distributed_2_bias:p
identity??dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_1_time_distributed_2_kerneldense_1_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_162486\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????ph
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?C
?
F__inference_sequential_layer_call_and_return_conditional_losses_163649

lstm_input-
lstm_lstm_lstm_cell_kernel:	?8
$lstm_lstm_lstm_cell_recurrent_kernel:
??'
lstm_lstm_lstm_cell_bias:	?<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?p8
*time_distributed_2_time_distributed_2_bias:p>
,time_distributed_4_time_distributed_4_kernel:p88
*time_distributed_4_time_distributed_4_bias:8>
,time_distributed_6_time_distributed_6_kernel:88
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_3/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_5/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_163481?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163323o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshape%lstm/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163296q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0,time_distributed_2_time_distributed_2_kernel*time_distributed_2_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163268q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0+^time_distributed_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163241q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_3/StatefulPartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163213q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape3time_distributed_3/StatefulPartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_5/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0+^time_distributed_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163186q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_5/StatefulPartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163158q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape3time_distributed_5/StatefulPartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_5/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_5/StatefulPartitionedCall*time_distributed_5/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?2
?
lstm_while_body_163758
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Z
add_1AddV2lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: ]

Identity_1Identitylstm_while_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0".
lstm_strided_slice_1lstm_strided_slice_1_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163241

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????pW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????p?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????p?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????p"
identityIdentity:output:0**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_164158

inputsM
:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?Z
Flstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??H
9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?X
Dtime_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??R
Ctime_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?]
Jtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pW
Itime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:p\
Jtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:p8W
Itime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:8\
Jtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:8W
Itime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??%lstm/lstm_cell/BiasAdd/ReadVariableOp?$lstm/lstm_cell/MatMul/ReadVariableOp?&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_2/dense_1/BiasAdd/ReadVariableOp?0time_distributed_2/dense_1/MatMul/ReadVariableOp?1time_distributed_4/dense_2/BiasAdd/ReadVariableOp?0time_distributed_4/dense_2/MatMul/ReadVariableOp?1time_distributed_6/dense_3/BiasAdd/ReadVariableOp?0time_distributed_6/dense_3/MatMul/ReadVariableOp@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splits
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????m
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????j
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_163977*"
condR
lstm_while_cond_163976*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshapelstm/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOpDtime_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpCtime_distributed_dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed/Reshape_1Reshape)time_distributed/dense/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????q
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/Reshape_2Reshapelstm/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????m
(time_distributed_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
&time_distributed_1/dropout/dropout/MulMul#time_distributed_1/Reshape:output:01time_distributed_1/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????{
(time_distributed_1/dropout/dropout/ShapeShape#time_distributed_1/Reshape:output:0*
T0*
_output_shapes
:?
?time_distributed_1/dropout/dropout/random_uniform/RandomUniformRandomUniform1time_distributed_1/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0v
1time_distributed_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
/time_distributed_1/dropout/dropout/GreaterEqualGreaterEqualHtime_distributed_1/dropout/dropout/random_uniform/RandomUniform:output:0:time_distributed_1/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
'time_distributed_1/dropout/dropout/CastCast3time_distributed_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
(time_distributed_1/dropout/dropout/Mul_1Mul*time_distributed_1/dropout/dropout/Mul:z:0+time_distributed_1/dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????w
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed_1/Reshape_1Reshape,time_distributed_1/dropout/dropout/Mul_1:z:0+time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????s
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
0time_distributed_2/dense_1/MatMul/ReadVariableOpReadVariableOpJtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
!time_distributed_2/dense_1/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
1time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
"time_distributed_2/dense_1/BiasAddBiasAdd+time_distributed_2/dense_1/MatMul:product:09time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
time_distributed_2/dense_1/ReluRelu+time_distributed_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
time_distributed_2/Reshape_1Reshape-time_distributed_2/dense_1/Relu:activations:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????ps
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????po
*time_distributed_3/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
(time_distributed_3/dropout_1/dropout/MulMul#time_distributed_3/Reshape:output:03time_distributed_3/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????p}
*time_distributed_3/dropout_1/dropout/ShapeShape#time_distributed_3/Reshape:output:0*
T0*
_output_shapes
:?
Atime_distributed_3/dropout_1/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_3/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0x
3time_distributed_3/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
1time_distributed_3/dropout_1/dropout/GreaterEqualGreaterEqualJtime_distributed_3/dropout_1/dropout/random_uniform/RandomUniform:output:0<time_distributed_3/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????p?
)time_distributed_3/dropout_1/dropout/CastCast5time_distributed_3/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????p?
*time_distributed_3/dropout_1/dropout/Mul_1Mul,time_distributed_3/dropout_1/dropout/Mul:z:0-time_distributed_3/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????pw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
time_distributed_3/Reshape_1Reshape.time_distributed_3/dropout_1/dropout/Mul_1:z:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????ps
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????pq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
0time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpJtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
!time_distributed_4/dense_2/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
1time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
"time_distributed_4/dense_2/BiasAddBiasAdd+time_distributed_4/dense_2/MatMul:product:09time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
time_distributed_4/dense_2/ReluRelu+time_distributed_4/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
time_distributed_4/Reshape_1Reshape-time_distributed_4/dense_2/Relu:activations:0+time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/Reshape_2Reshape%time_distributed_3/Reshape_1:output:0+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????pq
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8o
*time_distributed_5/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
(time_distributed_5/dropout_2/dropout/MulMul#time_distributed_5/Reshape:output:03time_distributed_5/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????8}
*time_distributed_5/dropout_2/dropout/ShapeShape#time_distributed_5/Reshape:output:0*
T0*
_output_shapes
:?
Atime_distributed_5/dropout_2/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_5/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0x
3time_distributed_5/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
1time_distributed_5/dropout_2/dropout/GreaterEqualGreaterEqualJtime_distributed_5/dropout_2/dropout/random_uniform/RandomUniform:output:0<time_distributed_5/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8?
)time_distributed_5/dropout_2/dropout/CastCast5time_distributed_5/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8?
*time_distributed_5/dropout_2/dropout/Mul_1Mul,time_distributed_5/dropout_2/dropout/Mul:z:0-time_distributed_5/dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8w
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
time_distributed_5/Reshape_1Reshape.time_distributed_5/dropout_2/dropout/Mul_1:z:0+time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
0time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpJtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
!time_distributed_6/dense_3/MatMulMatMul#time_distributed_6/Reshape:output:08time_distributed_6/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1time_distributed_6/dense_3/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
"time_distributed_6/dense_3/BiasAddBiasAdd+time_distributed_6/dense_3/MatMul:product:09time_distributed_6/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
"time_distributed_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ?
time_distributed_6/Reshape_1Reshape+time_distributed_6/dense_3/BiasAdd:output:0+time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????s
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????8s
cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            u
 cropping1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            u
 cropping1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
cropping1d/strided_sliceStridedSlice%time_distributed_6/Reshape_1:output:0'cropping1d/strided_slice/stack:output:0)cropping1d/strided_slice/stack_1:output:0)cropping1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_maskt
IdentityIdentity!cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_2/dense_1/BiasAdd/ReadVariableOp1^time_distributed_2/dense_1/MatMul/ReadVariableOp2^time_distributed_4/dense_2/BiasAdd/ReadVariableOp1^time_distributed_4/dense_2/MatMul/ReadVariableOp2^time_distributed_6/dense_3/BiasAdd/ReadVariableOp1^time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2f
1time_distributed_2/dense_1/BiasAdd/ReadVariableOp1time_distributed_2/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_2/dense_1/MatMul/ReadVariableOp0time_distributed_2/dense_1/MatMul/ReadVariableOp2f
1time_distributed_4/dense_2/BiasAdd/ReadVariableOp1time_distributed_4/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_4/dense_2/MatMul/ReadVariableOp0time_distributed_4/dense_2/MatMul/ReadVariableOp2f
1time_distributed_6/dense_3/BiasAdd/ReadVariableOp1time_distributed_6/dense_3/BiasAdd/ReadVariableOp2d
0time_distributed_6/dense_3/MatMul/ReadVariableOp0time_distributed_6/dense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165035

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_164248
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_164248___redundant_placeholder0.
*while_cond_164248___redundant_placeholder1.
*while_cond_164248___redundant_placeholder2.
*while_cond_164248___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_162673

inputs3
!dense_2_time_distributed_4_kernel:p8-
dense_2_time_distributed_4_bias:8
identity??dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_2_time_distributed_4_kerneldense_2_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_162631\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165295

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????8W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????8"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????8:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?I
?
@__inference_lstm_layer_call_and_return_conditional_losses_164619

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164535*
condR
while_cond_164534*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_164998

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_162775

inputsA
/matmul_readvariableop_time_distributed_6_kernel:8<
.biasadd_readvariableop_time_distributed_6_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165109

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????pW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????p?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????p?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????p\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????p"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165272

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????8\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????8"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????8:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165390

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162057

inputs

states
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
O
3__inference_time_distributed_3_layer_call_fn_165055

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_162559m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????p"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165134

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????pW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????p?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????p?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????p"
identityIdentity:output:0**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?C
?
F__inference_sequential_layer_call_and_return_conditional_losses_163543

inputs-
lstm_lstm_lstm_cell_kernel:	?8
$lstm_lstm_lstm_cell_recurrent_kernel:
??'
lstm_lstm_lstm_cell_bias:	?<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?p8
*time_distributed_2_time_distributed_2_bias:p>
,time_distributed_4_time_distributed_4_kernel:p88
*time_distributed_4_time_distributed_4_bias:8>
,time_distributed_6_time_distributed_6_kernel:88
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_3/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_5/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_163481?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163323o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshape%lstm/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163296q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0,time_distributed_2_time_distributed_2_kernel*time_distributed_2_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163268q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0+^time_distributed_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163241q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_3/StatefulPartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163213q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape3time_distributed_3/StatefulPartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_5/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0+^time_distributed_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163186q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_5/StatefulPartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163158q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape3time_distributed_5/StatefulPartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_5/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_5/StatefulPartitionedCall*time_distributed_5/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_5_layer_call_fn_165241

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_162704m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????8"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????8:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_162640

inputs3
!dense_2_time_distributed_4_kernel:p8-
dense_2_time_distributed_4_bias:8
identity??dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_2_time_distributed_4_kerneldense_2_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_162631\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?=
?
F__inference_sequential_layer_call_and_return_conditional_losses_163116

inputs-
lstm_lstm_lstm_cell_kernel:	?8
$lstm_lstm_lstm_cell_recurrent_kernel:
??'
lstm_lstm_lstm_cell_bias:	?<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?p8
*time_distributed_2_time_distributed_2_bias:p>
,time_distributed_4_time_distributed_4_kernel:p88
*time_distributed_4_time_distributed_4_bias:8>
,time_distributed_6_time_distributed_6_kernel:88
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_162986?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_163005o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
time_distributed/ReshapeReshape%lstm/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163019q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:0,time_distributed_2_time_distributed_2_kernel*time_distributed_2_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163037q
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
time_distributed_2/ReshapeReshape+time_distributed_1/PartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163051q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163069q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   ?
time_distributed_4/ReshapeReshape+time_distributed_3/PartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163083q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163100q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   ?
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_163113v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165418

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163186

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????8W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????8"
identityIdentity:output:0**
_input_shapes
:?????????8:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_165604

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_162552`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????p"
identityIdentity:output:0*&
_input_shapes
:?????????p:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165320

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????8W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????8"
identityIdentity:output:0**
_input_shapes
:?????????8:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_165554

inputsA
-matmul_readvariableop_time_distributed_kernel:
??;
,biasadd_readvariableop_time_distributed_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
lstm_while_cond_163976
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_13
/lstm_while_cond_163976___redundant_placeholder03
/lstm_while_cond_163976___redundant_placeholder13
/lstm_while_cond_163976___redundant_placeholder23
/lstm_while_cond_163976___redundant_placeholder3
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
(__inference_dense_3_layer_call_fn_165678

inputs+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_162775o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
*__inference_lstm_cell_layer_call_fn_165472

inputs
states_0
states_1(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162191p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165236

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   8   ?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????8e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?1
?
while_body_164678
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?I
?
@__inference_lstm_layer_call_and_return_conditional_losses_162986

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_162902*
condR
while_cond_162901*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165050

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
@__inference_lstm_layer_call_and_return_conditional_losses_162134

inputs2
lstm_cell_lstm_lstm_cell_kernel:	?=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??,
lstm_cell_lstm_lstm_cell_bias:	?
identity??!lstm_cell/StatefulPartitionedCall?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_lstm_lstm_cell_kernel)lstm_cell_lstm_lstm_cell_recurrent_kernellstm_cell_lstm_lstm_cell_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162057n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_lstm_lstm_cell_kernel)lstm_cell_lstm_lstm_cell_recurrent_kernellstm_cell_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_162068*
condR
while_cond_162067*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_162697

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????8[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????8"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????8:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164812

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_4_layer_call_fn_165162

inputs+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_163213s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_162582

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????pC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????p*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????po
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????pi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????pY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????p"
identityIdentity:output:0*&
_input_shapes
:?????????p:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_164190

inputs(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_163481t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163019

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????a
dropout/IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_3_layer_call_fn_165070

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163241s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????p22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_165588

inputs,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_162486o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_164166
inputs_0(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_162134}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_162748

inputs
identity??!dropout_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_162727\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape*dropout_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8j
NoOpNoOp"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????82F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_162727

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????8C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????8*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????8o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????8i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????8Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????8"
identityIdentity:output:0*&
_input_shapes
:?????????8:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_162704

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dropout_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_162697\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape"dropout_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????8"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????8:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_165348

inputs+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163158s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_165444

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_1_layer_call_fn_164884

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163296t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?J
?
@__inference_lstm_layer_call_and_return_conditional_losses_164333
inputs_0H
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164249*
condR
while_cond_164248*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?1
?
while_body_164392
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163037

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_162834

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'???????????????????????????*

begin_mask*
end_maskt
IdentityIdentitystrided_slice:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_162458

inputs
identity??dropout/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dropout/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_162437\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dropout/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????h
NoOpNoOp ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*4
_input_shapes#
!:???????????????????2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?8
?
@__inference_lstm_layer_call_and_return_conditional_losses_162310

inputs2
lstm_cell_lstm_lstm_cell_kernel:	?=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??,
lstm_cell_lstm_lstm_cell_bias:	?
identity??!lstm_cell/StatefulPartitionedCall?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_lstm_lstm_cell_kernel)lstm_cell_lstm_lstm_cell_recurrent_kernellstm_cell_lstm_lstm_cell_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_162191n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_lstm_lstm_cell_kernel)lstm_cell_lstm_lstm_cell_recurrent_kernellstm_cell_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_162244*
condR
while_cond_162243*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:??????????????????: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_163699

inputs(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_163543s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_163051

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????pb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????p^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????p"
identityIdentity:output:0**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_165633

inputs+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_162631o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_164182

inputs(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_162986t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_1_layer_call_fn_164879

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_163019e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_163667

lstm_input(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_161990s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?1
?
while_body_163397
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_164249
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_163158

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:8D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0?
dense_3/MatMulMatMulReshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????8: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164923

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??
dropout/dropout/MulMulReshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????U
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165086

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????pb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????p\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????p"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_165436

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'???????????????????????????*

begin_mask*
end_maskt
IdentityIdentitystrided_slice:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165184

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:p8D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:8
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????8\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :8?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_2/Relu:activations:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????8n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????8?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????p: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_165644

inputsA
/matmul_readvariableop_time_distributed_4_kernel:p8<
.biasadd_readvariableop_time_distributed_4_bias:8
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:p8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:8*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????8P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????8a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????8w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_165659

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????8[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????8"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????8:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_162817

inputs3
!dense_3_time_distributed_6_kernel:8-
dense_3_time_distributed_6_bias:
identity??dense_3/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????8   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????8?
dense_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_3_time_distributed_6_kerneldense_3_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_162775\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????h
NoOpNoOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????8: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????8
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164849

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_time_distributed_bias*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_165688

inputsA
/matmul_readvariableop_time_distributed_6_kernel:8<
.biasadd_readvariableop_time_distributed_6_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:8*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_6_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_1_layer_call_fn_164869

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_162414n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_5_layer_call_fn_165251

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_163083d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????8"
identityIdentity:output:0**
_input_shapes
:?????????8:S O
+
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_162559

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????p   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????p?
dropout_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_162552\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape"dropout_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????p"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????p:\ X
4
_output_shapes"
 :??????????????????p
 
_user_specified_nameinputs
?2
?
lstm_while_body_163977
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	??? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
lstm_cell/MatMul/ReadVariableOpReadVariableOp7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:???G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Z
add_1AddV2lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: ]

Identity_1Identitylstm_while_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: e

Identity_4Identitylstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"n
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0"?
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernelClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0"p
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0".
lstm_strided_slice_1lstm_strided_slice_1_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_sequential_layer_call_fn_163573

lstm_input(
lstm_lstm_cell_kernel:	?3
lstm_lstm_cell_recurrent_kernel:
??"
lstm_lstm_cell_bias:	?+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?p%
time_distributed_2_bias:p+
time_distributed_4_kernel:p8%
time_distributed_4_bias:8+
time_distributed_6_kernel:8%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_163543s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_162528

inputs4
!dense_1_time_distributed_2_kernel:	?p-
dense_1_time_distributed_2_bias:p
identity??dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_1_time_distributed_2_kerneldense_1_time_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_162486\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????pn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????ph
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*8
_input_shapes'
%:???????????????????: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
!sequential_lstm_while_cond_161829&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_sequential_lstm_strided_slice_1>
:sequential_lstm_while_cond_161829___redundant_placeholder0>
:sequential_lstm_while_cond_161829___redundant_placeholder1>
:sequential_lstm_while_cond_161829___redundant_placeholder2>
:sequential_lstm_while_cond_161829___redundant_placeholder3
identity
`
LessLessplaceholder$less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_162486

inputsB
/matmul_readvariableop_time_distributed_2_kernel:	?p<
.biasadd_readvariableop_time_distributed_2_bias:p
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????pP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????pa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
@__inference_lstm_layer_call_and_return_conditional_losses_163481

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163397*
condR
while_cond_163396*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_164534
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_164534___redundant_placeholder0.
*while_cond_164534___redundant_placeholder1.
*while_cond_164534___redundant_placeholder2.
*while_cond_164534___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_163268

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?pD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:p
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?p*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:p*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????pd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   p   ?
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
@__inference_lstm_layer_call_and_return_conditional_losses_164762

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????c
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:??????????~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernelAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164678*
condR
while_cond_164677*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_165581

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E

lstm_input7
serving_default_lstm_input:0?????????B

cropping1d4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
	layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	 layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	%layer
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	*layer
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	/layer
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	4layer
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
I
=iter
	>decay
?learning_rate
@momentum"
	optimizer
n
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10"
trackable_list_wrapper
n
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
Q
state_size

Akernel
Brecurrent_kernel
Cbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Dkernel
Ebias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Gbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/SGD/iter
: (2training/SGD/decay
$:" (2training/SGD/learning_rate
: (2training/SGD/momentum
(:&	?2lstm/lstm_cell/kernel
3:1
??2lstm/lstm_cell/recurrent_kernel
": ?2lstm/lstm_cell/bias
+:)
??2time_distributed/kernel
$:"?2time_distributed/bias
,:*	?p2time_distributed_2/kernel
%:#p2time_distributed_2/bias
+:)p82time_distributed_4/kernel
%:#82time_distributed_4/bias
+:)82time_distributed_6/kernel
%:#2time_distributed_6/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
+__inference_sequential_layer_call_fn_163130
+__inference_sequential_layer_call_fn_163683
+__inference_sequential_layer_call_fn_163699
+__inference_sequential_layer_call_fn_163573?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_163918
F__inference_sequential_layer_call_and_return_conditional_losses_164158
F__inference_sequential_layer_call_and_return_conditional_losses_163611
F__inference_sequential_layer_call_and_return_conditional_losses_163649?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_161990
lstm_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_lstm_layer_call_fn_164166
%__inference_lstm_layer_call_fn_164174
%__inference_lstm_layer_call_fn_164182
%__inference_lstm_layer_call_fn_164190?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_lstm_layer_call_and_return_conditional_losses_164333
@__inference_lstm_layer_call_and_return_conditional_losses_164476
@__inference_lstm_layer_call_and_return_conditional_losses_164619
@__inference_lstm_layer_call_and_return_conditional_losses_164762?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_time_distributed_layer_call_fn_164769
1__inference_time_distributed_layer_call_fn_164776
1__inference_time_distributed_layer_call_fn_164783
1__inference_time_distributed_layer_call_fn_164790?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164812
L__inference_time_distributed_layer_call_and_return_conditional_losses_164834
L__inference_time_distributed_layer_call_and_return_conditional_losses_164849
L__inference_time_distributed_layer_call_and_return_conditional_losses_164864?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_1_layer_call_fn_164869
3__inference_time_distributed_1_layer_call_fn_164874
3__inference_time_distributed_1_layer_call_fn_164879
3__inference_time_distributed_1_layer_call_fn_164884?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164900
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164923
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164932
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164948?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_2_layer_call_fn_164955
3__inference_time_distributed_2_layer_call_fn_164962
3__inference_time_distributed_2_layer_call_fn_164969
3__inference_time_distributed_2_layer_call_fn_164976?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_164998
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165020
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165035
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165050?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_3_layer_call_fn_165055
3__inference_time_distributed_3_layer_call_fn_165060
3__inference_time_distributed_3_layer_call_fn_165065
3__inference_time_distributed_3_layer_call_fn_165070?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165086
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165109
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165118
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165134?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_4_layer_call_fn_165141
3__inference_time_distributed_4_layer_call_fn_165148
3__inference_time_distributed_4_layer_call_fn_165155
3__inference_time_distributed_4_layer_call_fn_165162?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165184
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165206
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165221
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165236?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_5_layer_call_fn_165241
3__inference_time_distributed_5_layer_call_fn_165246
3__inference_time_distributed_5_layer_call_fn_165251
3__inference_time_distributed_5_layer_call_fn_165256?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165272
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165295
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165304
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165320?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_6_layer_call_fn_165327
3__inference_time_distributed_6_layer_call_fn_165334
3__inference_time_distributed_6_layer_call_fn_165341
3__inference_time_distributed_6_layer_call_fn_165348?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165369
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165390
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165404
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165418?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_cropping1d_layer_call_fn_165423
+__inference_cropping1d_layer_call_fn_165428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_cropping1d_layer_call_and_return_conditional_losses_165436
F__inference_cropping1d_layer_call_and_return_conditional_losses_165444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_163667
lstm_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_lstm_cell_layer_call_fn_165458
*__inference_lstm_cell_layer_call_fn_165472?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165504
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165536?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_165543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_165554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_165559
(__inference_dropout_layer_call_fn_165564?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_165569
C__inference_dropout_layer_call_and_return_conditional_losses_165581?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_165588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_165599?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_165604
*__inference_dropout_1_layer_call_fn_165609?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_165614
E__inference_dropout_1_layer_call_and_return_conditional_losses_165626?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_165633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_165644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_165649
*__inference_dropout_2_layer_call_fn_165654?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_165659
E__inference_dropout_2_layer_call_and_return_conditional_losses_165671?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_165678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_165688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_161990?ABCDEFGHIJK7?4
-?*
(?%

lstm_input?????????
? ";?8
6

cropping1d(?%

cropping1d??????????
F__inference_cropping1d_layer_call_and_return_conditional_losses_165436?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
F__inference_cropping1d_layer_call_and_return_conditional_losses_165444`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_cropping1d_layer_call_fn_165423wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
+__inference_cropping1d_layer_call_fn_165428S3?0
)?&
$?!
inputs?????????
? "???????????
C__inference_dense_1_layer_call_and_return_conditional_losses_165599]FG0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????p
? |
(__inference_dense_1_layer_call_fn_165588PFG0?-
&?#
!?
inputs??????????
? "??????????p?
C__inference_dense_2_layer_call_and_return_conditional_losses_165644\HI/?,
%?"
 ?
inputs?????????p
? "%?"
?
0?????????8
? {
(__inference_dense_2_layer_call_fn_165633OHI/?,
%?"
 ?
inputs?????????p
? "??????????8?
C__inference_dense_3_layer_call_and_return_conditional_losses_165688\JK/?,
%?"
 ?
inputs?????????8
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_165678OJK/?,
%?"
 ?
inputs?????????8
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_165554^DE0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_layer_call_fn_165543QDE0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_165614\3?0
)?&
 ?
inputs?????????p
p 
? "%?"
?
0?????????p
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_165626\3?0
)?&
 ?
inputs?????????p
p
? "%?"
?
0?????????p
? }
*__inference_dropout_1_layer_call_fn_165604O3?0
)?&
 ?
inputs?????????p
p 
? "??????????p}
*__inference_dropout_1_layer_call_fn_165609O3?0
)?&
 ?
inputs?????????p
p
? "??????????p?
E__inference_dropout_2_layer_call_and_return_conditional_losses_165659\3?0
)?&
 ?
inputs?????????8
p 
? "%?"
?
0?????????8
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_165671\3?0
)?&
 ?
inputs?????????8
p
? "%?"
?
0?????????8
? }
*__inference_dropout_2_layer_call_fn_165649O3?0
)?&
 ?
inputs?????????8
p 
? "??????????8}
*__inference_dropout_2_layer_call_fn_165654O3?0
)?&
 ?
inputs?????????8
p
? "??????????8?
C__inference_dropout_layer_call_and_return_conditional_losses_165569^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_165581^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_165559Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_165564Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165504?ABC??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_165536?ABC??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
*__inference_lstm_cell_layer_call_fn_165458?ABC??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
*__inference_lstm_cell_layer_call_fn_165472?ABC??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
@__inference_lstm_layer_call_and_return_conditional_losses_164333?ABCO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_164476?ABCO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_164619rABC??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_164762rABC??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
%__inference_lstm_layer_call_fn_164166~ABCO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
%__inference_lstm_layer_call_fn_164174~ABCO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
%__inference_lstm_layer_call_fn_164182eABC??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
%__inference_lstm_layer_call_fn_164190eABC??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
F__inference_sequential_layer_call_and_return_conditional_losses_163611yABCDEFGHIJK??<
5?2
(?%

lstm_input?????????
p 

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_163649yABCDEFGHIJK??<
5?2
(?%

lstm_input?????????
p

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_163918uABCDEFGHIJK;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_164158uABCDEFGHIJK;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
+__inference_sequential_layer_call_fn_163130lABCDEFGHIJK??<
5?2
(?%

lstm_input?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_163573lABCDEFGHIJK??<
5?2
(?%

lstm_input?????????
p

 
? "???????????
+__inference_sequential_layer_call_fn_163683hABCDEFGHIJK;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_163699hABCDEFGHIJK;?8
1?.
$?!
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_163667?ABCDEFGHIJKE?B
? 
;?8
6

lstm_input(?%

lstm_input?????????";?8
6

cropping1d(?%

cropping1d??????????
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164900|E?B
;?8
.?+
inputs???????????????????
p 

 
? "3?0
)?&
0???????????????????
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164923|E?B
;?8
.?+
inputs???????????????????
p

 
? "3?0
)?&
0???????????????????
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164932j<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_164948j<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
3__inference_time_distributed_1_layer_call_fn_164869oE?B
;?8
.?+
inputs???????????????????
p 

 
? "&?#????????????????????
3__inference_time_distributed_1_layer_call_fn_164874oE?B
;?8
.?+
inputs???????????????????
p

 
? "&?#????????????????????
3__inference_time_distributed_1_layer_call_fn_164879]<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
3__inference_time_distributed_1_layer_call_fn_164884]<?9
2?/
%?"
inputs??????????
p

 
? "????????????
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_164998FGE?B
;?8
.?+
inputs???????????????????
p 

 
? "2?/
(?%
0??????????????????p
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165020FGE?B
;?8
.?+
inputs???????????????????
p

 
? "2?/
(?%
0??????????????????p
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165035mFG<?9
2?/
%?"
inputs??????????
p 

 
? ")?&
?
0?????????p
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_165050mFG<?9
2?/
%?"
inputs??????????
p

 
? ")?&
?
0?????????p
? ?
3__inference_time_distributed_2_layer_call_fn_164955rFGE?B
;?8
.?+
inputs???????????????????
p 

 
? "%?"??????????????????p?
3__inference_time_distributed_2_layer_call_fn_164962rFGE?B
;?8
.?+
inputs???????????????????
p

 
? "%?"??????????????????p?
3__inference_time_distributed_2_layer_call_fn_164969`FG<?9
2?/
%?"
inputs??????????
p 

 
? "??????????p?
3__inference_time_distributed_2_layer_call_fn_164976`FG<?9
2?/
%?"
inputs??????????
p

 
? "??????????p?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165086zD?A
:?7
-?*
inputs??????????????????p
p 

 
? "2?/
(?%
0??????????????????p
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165109zD?A
:?7
-?*
inputs??????????????????p
p

 
? "2?/
(?%
0??????????????????p
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165118h;?8
1?.
$?!
inputs?????????p
p 

 
? ")?&
?
0?????????p
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_165134h;?8
1?.
$?!
inputs?????????p
p

 
? ")?&
?
0?????????p
? ?
3__inference_time_distributed_3_layer_call_fn_165055mD?A
:?7
-?*
inputs??????????????????p
p 

 
? "%?"??????????????????p?
3__inference_time_distributed_3_layer_call_fn_165060mD?A
:?7
-?*
inputs??????????????????p
p

 
? "%?"??????????????????p?
3__inference_time_distributed_3_layer_call_fn_165065[;?8
1?.
$?!
inputs?????????p
p 

 
? "??????????p?
3__inference_time_distributed_3_layer_call_fn_165070[;?8
1?.
$?!
inputs?????????p
p

 
? "??????????p?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165184~HID?A
:?7
-?*
inputs??????????????????p
p 

 
? "2?/
(?%
0??????????????????8
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165206~HID?A
:?7
-?*
inputs??????????????????p
p

 
? "2?/
(?%
0??????????????????8
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165221lHI;?8
1?.
$?!
inputs?????????p
p 

 
? ")?&
?
0?????????8
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_165236lHI;?8
1?.
$?!
inputs?????????p
p

 
? ")?&
?
0?????????8
? ?
3__inference_time_distributed_4_layer_call_fn_165141qHID?A
:?7
-?*
inputs??????????????????p
p 

 
? "%?"??????????????????8?
3__inference_time_distributed_4_layer_call_fn_165148qHID?A
:?7
-?*
inputs??????????????????p
p

 
? "%?"??????????????????8?
3__inference_time_distributed_4_layer_call_fn_165155_HI;?8
1?.
$?!
inputs?????????p
p 

 
? "??????????8?
3__inference_time_distributed_4_layer_call_fn_165162_HI;?8
1?.
$?!
inputs?????????p
p

 
? "??????????8?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165272zD?A
:?7
-?*
inputs??????????????????8
p 

 
? "2?/
(?%
0??????????????????8
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165295zD?A
:?7
-?*
inputs??????????????????8
p

 
? "2?/
(?%
0??????????????????8
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165304h;?8
1?.
$?!
inputs?????????8
p 

 
? ")?&
?
0?????????8
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_165320h;?8
1?.
$?!
inputs?????????8
p

 
? ")?&
?
0?????????8
? ?
3__inference_time_distributed_5_layer_call_fn_165241mD?A
:?7
-?*
inputs??????????????????8
p 

 
? "%?"??????????????????8?
3__inference_time_distributed_5_layer_call_fn_165246mD?A
:?7
-?*
inputs??????????????????8
p

 
? "%?"??????????????????8?
3__inference_time_distributed_5_layer_call_fn_165251[;?8
1?.
$?!
inputs?????????8
p 

 
? "??????????8?
3__inference_time_distributed_5_layer_call_fn_165256[;?8
1?.
$?!
inputs?????????8
p

 
? "??????????8?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165369~JKD?A
:?7
-?*
inputs??????????????????8
p 

 
? "2?/
(?%
0??????????????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165390~JKD?A
:?7
-?*
inputs??????????????????8
p

 
? "2?/
(?%
0??????????????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165404lJK;?8
1?.
$?!
inputs?????????8
p 

 
? ")?&
?
0?????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_165418lJK;?8
1?.
$?!
inputs?????????8
p

 
? ")?&
?
0?????????
? ?
3__inference_time_distributed_6_layer_call_fn_165327qJKD?A
:?7
-?*
inputs??????????????????8
p 

 
? "%?"???????????????????
3__inference_time_distributed_6_layer_call_fn_165334qJKD?A
:?7
-?*
inputs??????????????????8
p

 
? "%?"???????????????????
3__inference_time_distributed_6_layer_call_fn_165341_JK;?8
1?.
$?!
inputs?????????8
p 

 
? "???????????
3__inference_time_distributed_6_layer_call_fn_165348_JK;?8
1?.
$?!
inputs?????????8
p

 
? "???????????
L__inference_time_distributed_layer_call_and_return_conditional_losses_164812?DEE?B
;?8
.?+
inputs???????????????????
p 

 
? "3?0
)?&
0???????????????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164834?DEE?B
;?8
.?+
inputs???????????????????
p

 
? "3?0
)?&
0???????????????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164849nDE<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_164864nDE<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
1__inference_time_distributed_layer_call_fn_164769sDEE?B
;?8
.?+
inputs???????????????????
p 

 
? "&?#????????????????????
1__inference_time_distributed_layer_call_fn_164776sDEE?B
;?8
.?+
inputs???????????????????
p

 
? "&?#????????????????????
1__inference_time_distributed_layer_call_fn_164783aDE<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
1__inference_time_distributed_layer_call_fn_164790aDE<?9
2?/
%?"
inputs??????????
p

 
? "???????????