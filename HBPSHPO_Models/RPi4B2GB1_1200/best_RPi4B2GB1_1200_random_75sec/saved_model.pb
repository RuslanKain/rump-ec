??<
??
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
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??8
~
training/Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *&
shared_nametraining/Adagrad/iter
w
)training/Adagrad/iter/Read/ReadVariableOpReadVariableOptraining/Adagrad/iter*
_output_shapes
: *
dtype0	
?
training/Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining/Adagrad/decay
y
*training/Adagrad/decay/Read/ReadVariableOpReadVariableOptraining/Adagrad/decay*
_output_shapes
: *
dtype0
?
training/Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training/Adagrad/learning_rate
?
2training/Adagrad/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adagrad/learning_rate*
_output_shapes
: *
dtype0
?
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_namelstm/lstm_cell/kernel
?
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	?
*
dtype0
?
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*0
shared_name!lstm/lstm_cell/recurrent_kernel
?
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
??
*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:?
*
dtype0
?
lstm_1/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
**
shared_namelstm_1/lstm_cell_1/kernel
?
-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/kernel* 
_output_shapes
:
??
*
dtype0
?
#lstm_1/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*4
shared_name%#lstm_1/lstm_cell_1/recurrent_kernel
?
7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_1/recurrent_kernel* 
_output_shapes
:
??
*
dtype0
?
lstm_1/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*(
shared_namelstm_1/lstm_cell_1/bias
?
+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/bias*
_output_shapes	
:?
*
dtype0
?
lstm_2/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
**
shared_namelstm_2/lstm_cell_2/kernel
?
-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/kernel* 
_output_shapes
:
??
*
dtype0
?
#lstm_2/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*4
shared_name%#lstm_2/lstm_cell_2/recurrent_kernel
?
7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
??
*
dtype0
?
lstm_2/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*(
shared_namelstm_2/lstm_cell_2/bias
?
+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/bias*
_output_shapes	
:?
*
dtype0
?
time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nametime_distributed/kernel
?
+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel* 
_output_shapes
:
??*
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
shape:	?T**
shared_nametime_distributed_2/kernel
?
-time_distributed_2/kernel/Read/ReadVariableOpReadVariableOptime_distributed_2/kernel*
_output_shapes
:	?T*
dtype0
?
time_distributed_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*(
shared_nametime_distributed_2/bias

+time_distributed_2/bias/Read/ReadVariableOpReadVariableOptime_distributed_2/bias*
_output_shapes
:T*
dtype0
?
time_distributed_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T***
shared_nametime_distributed_4/kernel
?
-time_distributed_4/kernel/Read/ReadVariableOpReadVariableOptime_distributed_4/kernel*
_output_shapes

:T**
dtype0
?
time_distributed_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**(
shared_nametime_distributed_4/bias

+time_distributed_4/bias/Read/ReadVariableOpReadVariableOptime_distributed_4/bias*
_output_shapes
:**
dtype0
?
time_distributed_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:***
shared_nametime_distributed_6/kernel
?
-time_distributed_6/kernel/Read/ReadVariableOpReadVariableOptime_distributed_6/kernel*
_output_shapes

:**
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
?
2training/Adagrad/lstm/lstm_cell/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*C
shared_name42training/Adagrad/lstm/lstm_cell/kernel/accumulator
?
Ftraining/Adagrad/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpReadVariableOp2training/Adagrad/lstm/lstm_cell/kernel/accumulator*
_output_shapes
:	?
*
dtype0
?
<training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*M
shared_name><training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator
?
Ptraining/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOpReadVariableOp<training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator* 
_output_shapes
:
??
*
dtype0
?
0training/Adagrad/lstm/lstm_cell/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*A
shared_name20training/Adagrad/lstm/lstm_cell/bias/accumulator
?
Dtraining/Adagrad/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpReadVariableOp0training/Adagrad/lstm/lstm_cell/bias/accumulator*
_output_shapes	
:?
*
dtype0
?
6training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*G
shared_name86training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator
?
Jtraining/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp6training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator* 
_output_shapes
:
??
*
dtype0
?
@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*Q
shared_nameB@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator
?
Ttraining/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator/Read/ReadVariableOpReadVariableOp@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator* 
_output_shapes
:
??
*
dtype0
?
4training/Adagrad/lstm_1/lstm_cell_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*E
shared_name64training/Adagrad/lstm_1/lstm_cell_1/bias/accumulator
?
Htraining/Adagrad/lstm_1/lstm_cell_1/bias/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/lstm_1/lstm_cell_1/bias/accumulator*
_output_shapes	
:?
*
dtype0
?
6training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*G
shared_name86training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator
?
Jtraining/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp6training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator* 
_output_shapes
:
??
*
dtype0
?
@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*Q
shared_nameB@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator
?
Ttraining/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator/Read/ReadVariableOpReadVariableOp@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator* 
_output_shapes
:
??
*
dtype0
?
4training/Adagrad/lstm_2/lstm_cell_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*E
shared_name64training/Adagrad/lstm_2/lstm_cell_2/bias/accumulator
?
Htraining/Adagrad/lstm_2/lstm_cell_2/bias/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/lstm_2/lstm_cell_2/bias/accumulator*
_output_shapes	
:?
*
dtype0
?
4training/Adagrad/time_distributed/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*E
shared_name64training/Adagrad/time_distributed/kernel/accumulator
?
Htraining/Adagrad/time_distributed/kernel/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/time_distributed/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
2training/Adagrad/time_distributed/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42training/Adagrad/time_distributed/bias/accumulator
?
Ftraining/Adagrad/time_distributed/bias/accumulator/Read/ReadVariableOpReadVariableOp2training/Adagrad/time_distributed/bias/accumulator*
_output_shapes	
:?*
dtype0
?
6training/Adagrad/time_distributed_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?T*G
shared_name86training/Adagrad/time_distributed_2/kernel/accumulator
?
Jtraining/Adagrad/time_distributed_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp6training/Adagrad/time_distributed_2/kernel/accumulator*
_output_shapes
:	?T*
dtype0
?
4training/Adagrad/time_distributed_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*E
shared_name64training/Adagrad/time_distributed_2/bias/accumulator
?
Htraining/Adagrad/time_distributed_2/bias/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/time_distributed_2/bias/accumulator*
_output_shapes
:T*
dtype0
?
6training/Adagrad/time_distributed_4/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T**G
shared_name86training/Adagrad/time_distributed_4/kernel/accumulator
?
Jtraining/Adagrad/time_distributed_4/kernel/accumulator/Read/ReadVariableOpReadVariableOp6training/Adagrad/time_distributed_4/kernel/accumulator*
_output_shapes

:T**
dtype0
?
4training/Adagrad/time_distributed_4/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:**E
shared_name64training/Adagrad/time_distributed_4/bias/accumulator
?
Htraining/Adagrad/time_distributed_4/bias/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/time_distributed_4/bias/accumulator*
_output_shapes
:**
dtype0
?
6training/Adagrad/time_distributed_6/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**G
shared_name86training/Adagrad/time_distributed_6/kernel/accumulator
?
Jtraining/Adagrad/time_distributed_6/kernel/accumulator/Read/ReadVariableOpReadVariableOp6training/Adagrad/time_distributed_6/kernel/accumulator*
_output_shapes

:**
dtype0
?
4training/Adagrad/time_distributed_6/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64training/Adagrad/time_distributed_6/bias/accumulator
?
Htraining/Adagrad/time_distributed_6/bias/accumulator/Read/ReadVariableOpReadVariableOp4training/Adagrad/time_distributed_6/bias/accumulator*
_output_shapes
:*
dtype0

NoOpNoOp
?h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?h
value?hB?h B?g
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
]
	$layer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
]
	)layer
*	variables
+trainable_variables
,regularization_losses
-	keras_api
]
	.layer
/	variables
0trainable_variables
1regularization_losses
2	keras_api
]
	3layer
4	variables
5trainable_variables
6regularization_losses
7	keras_api
]
	8layer
9	variables
:trainable_variables
;regularization_losses
<	keras_api
]
	=layer
>	variables
?trainable_variables
@regularization_losses
A	keras_api
]
	Blayer
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?
Kiter
	Ldecay
Mlearning_rateNaccumulator?Oaccumulator?Paccumulator?Qaccumulator?Raccumulator?Saccumulator?Taccumulator?Uaccumulator?Vaccumulator?Waccumulator?Xaccumulator?Yaccumulator?Zaccumulator?[accumulator?\accumulator?]accumulator?^accumulator?
~
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
~
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
 
?
d
state_size

Nkernel
Orecurrent_kernel
Pbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
 

N0
O1
P2

N0
O1
P2
 
?

istates
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
?
o
state_size

Qkernel
Rrecurrent_kernel
Sbias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
 

Q0
R1
S2

Q0
R1
S2
 
?

tstates
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
?
z
state_size

Tkernel
Urecurrent_kernel
Vbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
 

T0
U1
V2

T0
U1
V2
 
?

states
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
l

Wkernel
Xbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

W0
X1

W0
X1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
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
*	variables
+trainable_variables
,regularization_losses
l

Ykernel
Zbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

Y0
Z1

Y0
Z1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
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
4	variables
5trainable_variables
6regularization_losses
l

[kernel
\bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

[0
\1

[0
\1
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
>	variables
?trainable_variables
@regularization_losses
l

]kernel
^bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

]0
^1

]0
^1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
TR
VARIABLE_VALUEtraining/Adagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtraining/Adagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEtraining/Adagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_1/lstm_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_1/lstm_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/lstm_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_2/lstm_cell_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_2/lstm_cell_2/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_2/lstm_cell_2/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEtime_distributed/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtime_distributed_2/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtime_distributed_2/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtime_distributed_4/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtime_distributed_4/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtime_distributed_6/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtime_distributed_6/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
7
	8

9
10

?0
?1
 
 
 

N0
O1
P2

N0
O1
P2
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
 

0
 
 
 
 

Q0
R1
S2

Q0
R1
S2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 

0
 
 
 
 

T0
U1
V2

T0
U1
V2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
{	variables
|trainable_variables
}regularization_losses
 
 

0
 
 
 

W0
X1

W0
X1
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
$0
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
)0
 
 
 

Y0
Z1

Y0
Z1
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
.0
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
30
 
 
 

[0
\1

[0
\1
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
80
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
=0
 
 
 

]0
^1

]0
^1
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
B0
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
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
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
?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE2training/Adagrad/lstm/lstm_cell/kernel/accumulatorLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulatorLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0training/Adagrad/lstm/lstm_cell/bias/accumulatorLvariables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulatorLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulatorLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/lstm_1/lstm_cell_1/bias/accumulatorLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulatorLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulatorLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/lstm_2/lstm_cell_2/bias/accumulatorLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/time_distributed/kernel/accumulatorLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2training/Adagrad/time_distributed/bias/accumulatorMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6training/Adagrad/time_distributed_2/kernel/accumulatorMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/time_distributed_2/bias/accumulatorMvariables/12/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6training/Adagrad/time_distributed_4/kernel/accumulatorMvariables/13/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/time_distributed_4/bias/accumulatorMvariables/14/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6training/Adagrad/time_distributed_6/kernel/accumulatorMvariables/15/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4training/Adagrad/time_distributed_6/bias/accumulatorMvariables/16/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biaslstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biastime_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/biastime_distributed_4/kerneltime_distributed_4/biastime_distributed_6/kerneltime_distributed_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_321584
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)training/Adagrad/iter/Read/ReadVariableOp*training/Adagrad/decay/Read/ReadVariableOp2training/Adagrad/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOp-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_1/bias/Read/ReadVariableOp-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_2/bias/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_2/kernel/Read/ReadVariableOp+time_distributed_2/bias/Read/ReadVariableOp-time_distributed_4/kernel/Read/ReadVariableOp+time_distributed_4/bias/Read/ReadVariableOp-time_distributed_6/kernel/Read/ReadVariableOp+time_distributed_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpFtraining/Adagrad/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpPtraining/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOpDtraining/Adagrad/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpJtraining/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator/Read/ReadVariableOpTtraining/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator/Read/ReadVariableOpHtraining/Adagrad/lstm_1/lstm_cell_1/bias/accumulator/Read/ReadVariableOpJtraining/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator/Read/ReadVariableOpTtraining/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator/Read/ReadVariableOpHtraining/Adagrad/lstm_2/lstm_cell_2/bias/accumulator/Read/ReadVariableOpHtraining/Adagrad/time_distributed/kernel/accumulator/Read/ReadVariableOpFtraining/Adagrad/time_distributed/bias/accumulator/Read/ReadVariableOpJtraining/Adagrad/time_distributed_2/kernel/accumulator/Read/ReadVariableOpHtraining/Adagrad/time_distributed_2/bias/accumulator/Read/ReadVariableOpJtraining/Adagrad/time_distributed_4/kernel/accumulator/Read/ReadVariableOpHtraining/Adagrad/time_distributed_4/bias/accumulator/Read/ReadVariableOpJtraining/Adagrad/time_distributed_6/kernel/accumulator/Read/ReadVariableOpHtraining/Adagrad/time_distributed_6/bias/accumulator/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
__inference__traced_save_325606
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining/Adagrad/itertraining/Adagrad/decaytraining/Adagrad/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biaslstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biastime_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/biastime_distributed_4/kerneltime_distributed_4/biastime_distributed_6/kerneltime_distributed_6/biastotalcounttotal_1count_12training/Adagrad/lstm/lstm_cell/kernel/accumulator<training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator0training/Adagrad/lstm/lstm_cell/bias/accumulator6training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator4training/Adagrad/lstm_1/lstm_cell_1/bias/accumulator6training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator4training/Adagrad/lstm_2/lstm_cell_2/bias/accumulator4training/Adagrad/time_distributed/kernel/accumulator2training/Adagrad/time_distributed/bias/accumulator6training/Adagrad/time_distributed_2/kernel/accumulator4training/Adagrad/time_distributed_2/bias/accumulator6training/Adagrad/time_distributed_4/kernel/accumulator4training/Adagrad/time_distributed_4/bias/accumulator6training/Adagrad/time_distributed_6/kernel/accumulator4training/Adagrad/time_distributed_6/bias/accumulator*5
Tin.
,2**
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
"__inference__traced_restore_325739??6
?
?
while_cond_319205
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_319205___redundant_placeholder0.
*while_cond_319205___redundant_placeholder1.
*while_cond_319205___redundant_placeholder2.
*while_cond_319205___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?2
?
lstm_2_while_body_322437
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_2_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_2_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
value	B :\
add_1AddV2lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: _

Identity_1Identitylstm_2_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_2_strided_slice_1lstm_2_strided_slice_1_0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"?
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?8
?
@__inference_lstm_layer_call_and_return_conditional_losses_318950

inputs2
lstm_cell_lstm_lstm_cell_kernel:	?
=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??
,
lstm_cell_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318831n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_318884*
condR
while_cond_318883*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r
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
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324907

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????*"
identityIdentity:output:0**
_input_shapes
:?????????*:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
lstm_1_while_cond_322301
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_15
1lstm_1_while_cond_322301___redundant_placeholder05
1lstm_1_while_cond_322301___redundant_placeholder15
1lstm_1_while_cond_322301___redundant_placeholder25
1lstm_1_while_cond_322301___redundant_placeholder3
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320832

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ~
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Te
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????T?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_323706
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323706___redundant_placeholder0.
*while_cond_323706___redundant_placeholder1.
*while_cond_323706___redundant_placeholder2.
*while_cond_323706___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?9
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_319594

inputs9
%lstm_cell_2_lstm_2_lstm_cell_2_kernel:
??
C
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
2
#lstm_cell_2_lstm_2_lstm_cell_2_bias:	?

identity??#lstm_cell_2/StatefulPartitionedCall?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%lstm_cell_2_lstm_2_lstm_cell_2_kernel/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel#lstm_cell_2_lstm_2_lstm_cell_2_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319475n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_cell_2_lstm_2_lstm_cell_2_kernel/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel#lstm_cell_2_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319528*
condR
while_cond_319527*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_325318

inputs+
time_distributed_kernel:
??$
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
A__inference_dense_layer_call_and_return_conditional_losses_319624p
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_320886

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   }
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320860

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
 *  ??
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
 *??L>?
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
valueB"????   ?   ?
	Reshape_1Reshapedropout/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_318883
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_318883___redundant_placeholder0.
*while_cond_318883___redundant_placeholder1.
*while_cond_318883___redundant_placeholder2.
*while_cond_318883___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324790

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
*__inference_lstm_cell_layer_call_fn_325075

inputs
states_0
states_1(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318831p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
lstm_while_cond_321684
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_13
/lstm_while_cond_321684___redundant_placeholder03
/lstm_while_cond_321684___redundant_placeholder13
/lstm_while_cond_321684___redundant_placeholder23
/lstm_while_cond_321684___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
F
*__inference_dropout_1_layer_call_fn_325377

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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_319834`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????T"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_322637

inputs(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_320263t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_319666

inputs1
dense_time_distributed_kernel:
??*
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
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
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
A__inference_dense_layer_call_and_return_conditional_losses_319624\
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
%:???????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_320323
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_320323___redundant_placeholder0.
*while_cond_320323___redundant_placeholder1.
*while_cond_320323___redundant_placeholder2.
*while_cond_320323___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324511

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
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_322110

inputsM
:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
Z
Flstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
H
9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
V
Blstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
b
Nlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
P
Alstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
V
Blstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
b
Nlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
P
Alstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
X
Dtime_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??R
Ctime_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?]
Jtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TW
Itime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:T\
Jtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:T*W
Itime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:*\
Jtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:*W
Itime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??%lstm/lstm_cell/BiasAdd/ReadVariableOp?$lstm/lstm_cell/MatMul/ReadVariableOp?&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/while?)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp?(lstm_1/lstm_cell_1/MatMul/ReadVariableOp?*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp?lstm_1/while?)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?lstm_2/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_2/dense_1/BiasAdd/ReadVariableOp?0time_distributed_2/dense_1/MatMul/ReadVariableOp?1time_distributed_4/dense_2/BiasAdd/ReadVariableOp?0time_distributed_4/dense_2/MatMul/ReadVariableOp?1time_distributed_6/dense_3/BiasAdd/ReadVariableOp?0time_distributed_6/dense_3/MatMul/ReadVariableOp@

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
B :??
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
:??????????X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
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
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????N
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
:	?
*
dtype0?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splits
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_321685*"
condR
lstm_while_cond_321684*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
lstm_1/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose	Transposelstm/transpose_1:y:0lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
(lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpBlstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:00lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpNlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/zeros:output:02lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/MatMul:product:0%lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpAlstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_1/lstm_cell_1/BiasAddBiasAddlstm_1/lstm_cell_1/add:z:01lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
d
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0#lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split{
lstm_1/lstm_cell_1/SigmoidSigmoid!lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????}
lstm_1/lstm_cell_1/Sigmoid_1Sigmoid!lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mulMul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mul_1Mullstm_1/lstm_cell_1/Sigmoid:y:0!lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/add_1AddV2lstm_1/lstm_cell_1/mul:z:0lstm_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????}
lstm_1/lstm_cell_1/Sigmoid_2Sigmoid!lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mul_2Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Blstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelNlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelAlstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_321820*$
condR
lstm_1_while_cond_321819*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_2/ShapeShapelstm_1/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose	Transposelstm_1/transpose_1:y:0lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpBlstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpNlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAlstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
d
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split{
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????}
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????}
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Blstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelNlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelAlstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_321955*$
condR
lstm_2_while_cond_321954*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshapelstm_2/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOpDtime_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????q
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/Reshape_2Reshapelstm_2/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
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
valueB"????   ?   ?
time_distributed_1/Reshape_1Reshape,time_distributed_1/dropout/Identity:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????s
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
:	?T*
dtype0?
!time_distributed_2/dense_1/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
1time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
"time_distributed_2/dense_1/BiasAddBiasAdd+time_distributed_2/dense_1/MatMul:product:09time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Tw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
time_distributed_2/Reshape_1Reshape+time_distributed_2/dense_1/BiasAdd:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Ts
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
valueB"????T   ?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
%time_distributed_3/dropout_1/IdentityIdentity#time_distributed_3/Reshape:output:0*
T0*'
_output_shapes
:?????????Tw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
time_distributed_3/Reshape_1Reshape.time_distributed_3/dropout_1/Identity:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Ts
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????Tq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
0time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpJtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
!time_distributed_4/dense_2/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
1time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
"time_distributed_4/dense_2/BiasAddBiasAdd+time_distributed_4/dense_2/MatMul:product:09time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
time_distributed_4/Reshape_1Reshape+time_distributed_4/dense_2/BiasAdd:output:0+time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/Reshape_2Reshape%time_distributed_3/Reshape_1:output:0+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????Tq
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
%time_distributed_5/dropout_2/IdentityIdentity#time_distributed_5/Reshape:output:0*
T0*'
_output_shapes
:?????????*w
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
time_distributed_5/Reshape_1Reshape.time_distributed_5/dropout_2/Identity:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
0time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpJtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ?
time_distributed_6/Reshape_1Reshape+time_distributed_6/dense_3/BiasAdd:output:0+time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????s
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*s
cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           u
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
:?????????*

begin_mask*
end_maskt
IdentityIdentity!cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_1/MatMul/ReadVariableOp+^lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_2/dense_1/BiasAdd/ReadVariableOp1^time_distributed_2/dense_1/MatMul/ReadVariableOp2^time_distributed_4/dense_2/BiasAdd/ReadVariableOp1^time_distributed_4/dense_2/MatMul/ReadVariableOp2^time_distributed_6/dense_3/BiasAdd/ReadVariableOp1^time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2V
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2T
(lstm_1/lstm_cell_1/MatMul/ReadVariableOp(lstm_1/lstm_cell_1/MatMul/ReadVariableOp2X
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while2^
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
:?????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320581

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
valueB"????   ?   ?
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320643

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????*"
identityIdentity:output:0**
_input_shapes
:?????????*:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
while_cond_324294
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_324294___redundant_placeholder0.
*while_cond_324294___redundant_placeholder1.
*while_cond_324294___redundant_placeholder2.
*while_cond_324294___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324377

inputsO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_324295*
condR
while_cond_324294*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_323877
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323877___redundant_placeholder0.
*while_cond_323877___redundant_placeholder1.
*while_cond_323877___redundant_placeholder2.
*while_cond_323877___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
(__inference_dense_2_layer_call_fn_325406

inputs+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_319912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?9
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_319272

inputs9
%lstm_cell_1_lstm_1_lstm_cell_1_kernel:
??
C
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
2
#lstm_cell_1_lstm_1_lstm_cell_1_bias:	?

identity??#lstm_cell_1/StatefulPartitionedCall?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%lstm_cell_1_lstm_1_lstm_cell_1_kernel/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel#lstm_cell_1_lstm_1_lstm_cell_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319153n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_cell_1_lstm_1_lstm_cell_1_kernel/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel#lstm_cell_1_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319206*
condR
while_cond_319205*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
#sequential_lstm_2_while_cond_318478(
$sequential_lstm_2_while_loop_counter.
*sequential_lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_sequential_lstm_2_strided_slice_1@
<sequential_lstm_2_while_cond_318478___redundant_placeholder0@
<sequential_lstm_2_while_cond_318478___redundant_placeholder1@
<sequential_lstm_2_while_cond_318478___redundant_placeholder2@
<sequential_lstm_2_while_cond_318478___redundant_placeholder3
identity
b
LessLessplaceholder&less_sequential_lstm_2_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?1
?
while_body_320324
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_324156
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
lstm_2_while_cond_321954
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_2_strided_slice_15
1lstm_2_while_cond_321954___redundant_placeholder05
1lstm_2_while_cond_321954___redundant_placeholder15
1lstm_2_while_cond_321954___redundant_placeholder25
1lstm_2_while_cond_321954___redundant_placeholder3
identity
W
LessLessplaceholderless_lstm_2_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?0
?
while_body_322702
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324716

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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????TW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????T"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?0
?
while_body_321268
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_320115

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
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
?
?
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325281

inputs
states_0
states_1C
/matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
O
;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
*__inference_lstm_cell_layer_call_fn_325061

inputs
states_0
states_1(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318699p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325007

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
(__inference_dense_3_layer_call_fn_325450

inputs+
time_distributed_6_kernel:*%
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
C__inference_dense_3_layer_call_and_return_conditional_losses_320056o
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
:?????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
'__inference_lstm_1_layer_call_fn_323209
inputs_0-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_319098}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
1__inference_time_distributed_layer_call_fn_324384

inputs+
time_distributed_kernel:
??$
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_319633}
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
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324475

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   }
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_319921

inputs3
!dense_2_time_distributed_4_kernel:T*-
dense_2_time_distributed_4_bias:*
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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_2_time_distributed_4_kerneldense_2_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_319912\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
1__inference_time_distributed_layer_call_fn_324405

inputs+
time_distributed_kernel:
??$
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320886t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325311

inputs
states_0
states_1C
/matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
O
;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?1
?
while_body_321113
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_time_distributed_4_layer_call_fn_324755

inputs+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_319954|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_325460

inputsA
/matmul_readvariableop_time_distributed_6_kernel:*<
.biasadd_readvariableop_time_distributed_6_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
:?????????*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?1
?
while_body_323290
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?2
?
lstm_2_while_body_321955
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_2_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_2_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
value	B :\
add_1AddV2lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: _

Identity_1Identitylstm_2_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_2_strided_slice_1lstm_2_strided_slice_1_0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"?
Rtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_322979
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_322979___redundant_placeholder0.
*while_cond_322979___redundant_placeholder1.
*while_cond_322979___redundant_placeholder2.
*while_cond_322979___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?0
?
while_body_323119
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320612

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????Tb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????T"
identityIdentity:output:0**
_input_shapes
:?????????T:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320778

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ~
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_325328

inputsA
-matmul_readvariableop_time_distributed_kernel:
??;
,biasadd_readvariableop_time_distributed_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_321584

lstm_input(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?
-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?
-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?
+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*+
time_distributed_6_kernel:*%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biaslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_biaslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_318634s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_319864

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????TC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????To
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????Ti
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????TY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????T"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320065

inputs3
!dense_3_time_distributed_6_kernel:*-
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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
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
C__inference_dense_3_layer_call_and_return_conditional_losses_320056\
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
$:??????????????????*: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?I
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323372
inputs_0O
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while=
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323290*
condR
while_cond_323289*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_322784
inputs_0H
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322702*
condR
while_cond_322701*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
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
?
while_body_320958
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_time_distributed_2_layer_call_fn_324566

inputs,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_319777|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????T`
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
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_325399

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????TC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????To
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????Ti
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????TY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????T"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?9
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_319098

inputs9
%lstm_cell_1_lstm_1_lstm_cell_1_kernel:
??
C
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
2
#lstm_cell_1_lstm_1_lstm_cell_1_bias:	?

identity??#lstm_cell_1/StatefulPartitionedCall?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%lstm_cell_1_lstm_1_lstm_cell_1_kernel/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel#lstm_cell_1_lstm_1_lstm_cell_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319021n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_cell_1_lstm_1_lstm_cell_1_kernel/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel#lstm_cell_1_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319032*
condR
while_cond_319031*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_321267
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_321267___redundant_placeholder0.
*while_cond_321267___redundant_placeholder1.
*while_cond_321267___redundant_placeholder2.
*while_cond_321267___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?0
?
while_body_322980
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_time_distributed_3_layer_call_fn_324662

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
 :??????????????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_319841m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????T"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
while_cond_320957
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_320957___redundant_placeholder0.
*while_cond_320957___redundant_placeholder1.
*while_cond_320957___redundant_placeholder2.
*while_cond_320957___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_320567

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   }
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_lstm_2_layer_call_fn_323821

inputs-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_321040t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320029

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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_320008\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape*dropout_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*j
NoOpNoOp"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?
?
,__inference_lstm_cell_1_layer_call_fn_325163

inputs
states_0
states_1-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319153p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
O
3__inference_time_distributed_1_layer_call_fn_324490

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320581e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
while_body_320181
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
l
3__inference_time_distributed_3_layer_call_fn_324667

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
 :??????????????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_319885|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_322629
inputs_0(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_318950}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
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
while_cond_320466
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_320466___redundant_placeholder0.
*while_cond_320466___redundant_placeholder1.
*while_cond_320466___redundant_placeholder2.
*while_cond_320466___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?J
?

F__inference_sequential_layer_call_and_return_conditional_losses_321514

lstm_input-
lstm_lstm_lstm_cell_kernel:	?
8
$lstm_lstm_lstm_cell_recurrent_kernel:
??
'
lstm_lstm_lstm_cell_bias:	?
4
 lstm_1_lstm_1_lstm_cell_1_kernel:
??
>
*lstm_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
-
lstm_1_lstm_1_lstm_cell_1_bias:	?
4
 lstm_2_lstm_2_lstm_cell_2_kernel:
??
>
*lstm_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
-
lstm_2_lstm_2_lstm_cell_2_bias:	?
<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?T8
*time_distributed_2_time_distributed_2_bias:T>
,time_distributed_4_time_distributed_4_kernel:T*8
*time_distributed_4_time_distributed_4_bias:*>
,time_distributed_6_time_distributed_6_kernel:*8
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_320263?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 lstm_1_lstm_1_lstm_cell_1_kernel*lstm_1_lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_320406?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 lstm_2_lstm_2_lstm_cell_2_kernel*lstm_2_lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_320549?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320567o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshape'lstm_2/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320581q
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320598q
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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320612q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320629q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape+time_distributed_3/PartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320643q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320660q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?0
?
lstm_while_body_322167
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_324016
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_324016___redundant_placeholder0.
*while_cond_324016___redundant_placeholder1.
*while_cond_324016___redundant_placeholder2.
*while_cond_324016___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
lstm_2_while_cond_322436
lstm_2_while_loop_counter#
lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_2_strided_slice_15
1lstm_2_while_cond_322436___redundant_placeholder05
1lstm_2_while_cond_322436___redundant_placeholder15
1lstm_2_while_cond_322436___redundant_placeholder25
1lstm_2_while_cond_322436___redundant_placeholder3
identity
W
LessLessplaceholderless_lstm_2_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
!sequential_lstm_while_cond_318208&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_sequential_lstm_strided_slice_1>
:sequential_lstm_while_cond_318208___redundant_placeholder0>
:sequential_lstm_while_cond_318208___redundant_placeholder1>
:sequential_lstm_while_cond_318208___redundant_placeholder2>
:sequential_lstm_while_cond_318208___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324972

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
$:??????????????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?
?
while_cond_323567
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323567___redundant_placeholder0.
*while_cond_323567___redundant_placeholder1.
*while_cond_323567___redundant_placeholder2.
*while_cond_323567___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_319697

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
C__inference_dropout_layer_call_and_return_conditional_losses_319690\
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
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_320263

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320181*
condR
while_cond_320180*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_322621
inputs_0(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_318776}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
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
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_319768

inputsB
/matmul_readvariableop_time_distributed_2_kernel:	?T<
.biasadd_readvariableop_time_distributed_2_bias:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?T*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Tw
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
B__inference_lstm_1_layer_call_and_return_conditional_losses_323511
inputs_0O
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while=
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323429*
condR
while_cond_323428*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
3__inference_time_distributed_6_layer_call_fn_324944

inputs+
time_distributed_6_kernel:*%
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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320660s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_319834

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????T[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????T"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_324937

inputs+
time_distributed_6_kernel:*%
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
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320098|
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
$:??????????????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_321040

inputsO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320958*
condR
while_cond_320957*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_321606

inputs(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?
-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?
-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?
+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*+
time_distributed_6_kernel:*%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biaslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_biaslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_320676s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324923

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????*W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????*"
identityIdentity:output:0**
_input_shapes
:?????????*:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319153

inputs

states
states_1C
/matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
O
;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325135

inputs
states_0
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?
K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
O
3__inference_time_distributed_5_layer_call_fn_324854

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
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320643d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????*"
identityIdentity:output:0**
_input_shapes
:?????????*:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324657

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ~
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Te
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????T?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_lstm_2_layer_call_fn_323813

inputs-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_320549t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_body_318710
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!lstm_cell_lstm_lstm_cell_kernel_0:	?
?
+lstm_cell_lstm_lstm_cell_recurrent_kernel_0:
??
.
lstm_cell_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
lstm_cell_lstm_lstm_cell_kernel:	?
=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??
,
lstm_cell_lstm_lstm_cell_bias:	?
??!lstm_cell/StatefulPartitionedCall?
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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318699?
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
:??????????|

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????j
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
:: : : : :??????????:??????????: : : : : 2F
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318831

inputs

states
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?
K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_323428
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323428___redundant_placeholder0.
*while_cond_323428___redundant_placeholder1.
*while_cond_323428___redundant_placeholder2.
*while_cond_323428___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
? 
?
while_body_319354
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
'lstm_cell_2_lstm_2_lstm_cell_2_kernel_0:
??
E
1lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
4
%lstm_cell_2_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
%lstm_cell_2_lstm_2_lstm_cell_2_kernel:
??
C
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
2
#lstm_cell_2_lstm_2_lstm_cell_2_bias:	?
??#lstm_cell_2/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'lstm_cell_2_lstm_2_lstm_cell_2_kernel_01lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0%lstm_cell_2_lstm_2_lstm_cell_2_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319343?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_2/StatefulPartitionedCall:output:0*
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
: ~

Identity_4Identity,lstm_cell_2/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_5Identity,lstm_cell_2/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"L
#lstm_cell_2_lstm_2_lstm_cell_2_bias%lstm_cell_2_lstm_2_lstm_cell_2_bias_0"P
%lstm_cell_2_lstm_2_lstm_cell_2_kernel'lstm_cell_2_lstm_2_lstm_cell_2_kernel_0"d
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel1lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_325387

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????T[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????T"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_322923
inputs_0H
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322841*
condR
while_cond_322840*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
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
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324629

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????T?
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
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_319777

inputs4
!dense_1_time_distributed_2_kernel:	?T-
dense_1_time_distributed_2_bias:T
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_319768\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????Th
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
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324559

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
 *  ??
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
 *??L>?
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
valueB"????   ?   ?
	Reshape_1Reshapedropout/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324608

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????T?
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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324725

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????Tb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????T"
identityIdentity:output:0**
_input_shapes
:?????????T:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_321468

lstm_input(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?
-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?
-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?
+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*+
time_distributed_6_kernel:*%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biaslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_biaslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_321426s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?I
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_321195

inputsO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_321113*
condR
while_cond_321112*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_319978

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????*[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????*"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????*:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_325362

inputs,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_319768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????T`
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
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_319633

inputs1
dense_time_distributed_kernel:
??*
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
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
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
A__inference_dense_layer_call_and_return_conditional_losses_319624\
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
%:???????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
#sequential_lstm_1_while_cond_318343(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_sequential_lstm_1_strided_slice_1@
<sequential_lstm_1_while_cond_318343___redundant_placeholder0@
<sequential_lstm_1_while_cond_318343___redundant_placeholder1@
<sequential_lstm_1_while_cond_318343___redundant_placeholder2@
<sequential_lstm_1_while_cond_318343___redundant_placeholder3
identity
b
LessLessplaceholder&less_sequential_lstm_1_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320598

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ~
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Te
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????T?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_325338

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
C__inference_dropout_layer_call_and_return_conditional_losses_319720p
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
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_325047

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
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
:?????????*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?]
?
__inference__traced_save_325606
file_prefix4
0savev2_training_adagrad_iter_read_readvariableop	5
1savev2_training_adagrad_decay_read_readvariableop=
9savev2_training_adagrad_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop8
4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableopB
>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop6
2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop6
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
"savev2_count_1_read_readvariableopQ
Msavev2_training_adagrad_lstm_lstm_cell_kernel_accumulator_read_readvariableop[
Wsavev2_training_adagrad_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopO
Ksavev2_training_adagrad_lstm_lstm_cell_bias_accumulator_read_readvariableopU
Qsavev2_training_adagrad_lstm_1_lstm_cell_1_kernel_accumulator_read_readvariableop_
[savev2_training_adagrad_lstm_1_lstm_cell_1_recurrent_kernel_accumulator_read_readvariableopS
Osavev2_training_adagrad_lstm_1_lstm_cell_1_bias_accumulator_read_readvariableopU
Qsavev2_training_adagrad_lstm_2_lstm_cell_2_kernel_accumulator_read_readvariableop_
[savev2_training_adagrad_lstm_2_lstm_cell_2_recurrent_kernel_accumulator_read_readvariableopS
Osavev2_training_adagrad_lstm_2_lstm_cell_2_bias_accumulator_read_readvariableopS
Osavev2_training_adagrad_time_distributed_kernel_accumulator_read_readvariableopQ
Msavev2_training_adagrad_time_distributed_bias_accumulator_read_readvariableopU
Qsavev2_training_adagrad_time_distributed_2_kernel_accumulator_read_readvariableopS
Osavev2_training_adagrad_time_distributed_2_bias_accumulator_read_readvariableopU
Qsavev2_training_adagrad_time_distributed_4_kernel_accumulator_read_readvariableopS
Osavev2_training_adagrad_time_distributed_4_bias_accumulator_read_readvariableopU
Qsavev2_training_adagrad_time_distributed_6_kernel_accumulator_read_readvariableopS
Osavev2_training_adagrad_time_distributed_6_bias_accumulator_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/12/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/13/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/14/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/15/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/16/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_training_adagrad_iter_read_readvariableop1savev2_training_adagrad_decay_read_readvariableop9savev2_training_adagrad_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableop>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_2_kernel_read_readvariableop2savev2_time_distributed_2_bias_read_readvariableop4savev2_time_distributed_4_kernel_read_readvariableop2savev2_time_distributed_4_bias_read_readvariableop4savev2_time_distributed_6_kernel_read_readvariableop2savev2_time_distributed_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopMsavev2_training_adagrad_lstm_lstm_cell_kernel_accumulator_read_readvariableopWsavev2_training_adagrad_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopKsavev2_training_adagrad_lstm_lstm_cell_bias_accumulator_read_readvariableopQsavev2_training_adagrad_lstm_1_lstm_cell_1_kernel_accumulator_read_readvariableop[savev2_training_adagrad_lstm_1_lstm_cell_1_recurrent_kernel_accumulator_read_readvariableopOsavev2_training_adagrad_lstm_1_lstm_cell_1_bias_accumulator_read_readvariableopQsavev2_training_adagrad_lstm_2_lstm_cell_2_kernel_accumulator_read_readvariableop[savev2_training_adagrad_lstm_2_lstm_cell_2_recurrent_kernel_accumulator_read_readvariableopOsavev2_training_adagrad_lstm_2_lstm_cell_2_bias_accumulator_read_readvariableopOsavev2_training_adagrad_time_distributed_kernel_accumulator_read_readvariableopMsavev2_training_adagrad_time_distributed_bias_accumulator_read_readvariableopQsavev2_training_adagrad_time_distributed_2_kernel_accumulator_read_readvariableopOsavev2_training_adagrad_time_distributed_2_bias_accumulator_read_readvariableopQsavev2_training_adagrad_time_distributed_4_kernel_accumulator_read_readvariableopOsavev2_training_adagrad_time_distributed_4_bias_accumulator_read_readvariableopQsavev2_training_adagrad_time_distributed_6_kernel_accumulator_read_readvariableopOsavev2_training_adagrad_time_distributed_6_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : :	?
:
??
:?
:
??
:
??
:?
:
??
:
??
:?
:
??:?:	?T:T:T*:*:*:: : : : :	?
:
??
:?
:
??
:
??
:?
:
??
:
??
:?
:
??:?:	?T:T:T*:*:*:: 2(
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
: :%!

_output_shapes
:	?
:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:!	

_output_shapes	
:?
:&
"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?T: 

_output_shapes
:T:$ 

_output_shapes

:T*: 

_output_shapes
:*:$ 

_output_shapes

:*: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?
:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??
:&"
 
_output_shapes
:
??
:!

_output_shapes	
:?
:&"
 
_output_shapes
:
??
:& "
 
_output_shapes
:
??
:!!

_output_shapes	
:?
:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?T: %

_output_shapes
:T:$& 

_output_shapes

:T*: '

_output_shapes
:*:$( 

_output_shapes

:*: )

_output_shapes
::*

_output_shapes
: 
?
?
'__inference_lstm_1_layer_call_fn_323217
inputs_0-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_319272}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324741

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????TW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????T"
identityIdentity:output:0**
_input_shapes
:?????????T:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324693

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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????Tb
dropout_1/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????T\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_1/Identity:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????T"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_1_layer_call_fn_324485

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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_319741}
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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_319841

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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dropout_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_319834\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape"dropout_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????T"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_322613

inputsM
:lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
Z
Flstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
H
9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
V
Blstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
b
Nlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
P
Alstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
V
Blstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
b
Nlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
P
Alstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
X
Dtime_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??R
Ctime_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?]
Jtime_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TW
Itime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:T\
Jtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:T*W
Itime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:*\
Jtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:*W
Itime_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??%lstm/lstm_cell/BiasAdd/ReadVariableOp?$lstm/lstm_cell/MatMul/ReadVariableOp?&lstm/lstm_cell/MatMul_1/ReadVariableOp?
lstm/while?)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp?(lstm_1/lstm_cell_1/MatMul/ReadVariableOp?*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp?lstm_1/while?)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?lstm_2/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_2/dense_1/BiasAdd/ReadVariableOp?0time_distributed_2/dense_1/MatMul/ReadVariableOp?1time_distributed_4/dense_2/BiasAdd/ReadVariableOp?0time_distributed_4/dense_2/MatMul/ReadVariableOp?1time_distributed_6/dense_3/BiasAdd/ReadVariableOp?0time_distributed_6/dense_3/MatMul/ReadVariableOp@

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
B :??
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
:??????????X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
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
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????N
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
:	?
*
dtype0?
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpFlstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splits
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????u
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_322167*"
condR
lstm_while_cond_322166*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
lstm_1/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose	Transposelstm/transpose_1:y:0lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
(lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpBlstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:00lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpNlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/zeros:output:02lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/MatMul:product:0%lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpAlstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_1/lstm_cell_1/BiasAddBiasAddlstm_1/lstm_cell_1/add:z:01lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
d
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0#lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split{
lstm_1/lstm_cell_1/SigmoidSigmoid!lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????}
lstm_1/lstm_cell_1/Sigmoid_1Sigmoid!lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mulMul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mul_1Mullstm_1/lstm_cell_1/Sigmoid:y:0!lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/add_1AddV2lstm_1/lstm_cell_1/mul:z:0lstm_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????}
lstm_1/lstm_cell_1/Sigmoid_2Sigmoid!lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:???????????
lstm_1/lstm_cell_1/mul_2Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Blstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelNlstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelAlstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_322302*$
condR
lstm_1_while_cond_322301*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_2/ShapeShapelstm_1/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose	Transposelstm_1/transpose_1:y:0lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpBlstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpNlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAlstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
d
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split{
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????}
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????}
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Blstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelNlstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelAlstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_322437*$
condR
lstm_2_while_cond_322436*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshapelstm_2/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOpDtime_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????q
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/Reshape_2Reshapelstm_2/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????q
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
 *  ???
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
 *??L>?
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
valueB"????   ?   ?
time_distributed_1/Reshape_1Reshape,time_distributed_1/dropout/dropout/Mul_1:z:0+time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????s
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
:	?T*
dtype0?
!time_distributed_2/dense_1/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
1time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
"time_distributed_2/dense_1/BiasAddBiasAdd+time_distributed_2/dense_1/MatMul:product:09time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Tw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
time_distributed_2/Reshape_1Reshape+time_distributed_2/dense_1/BiasAdd:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Ts
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
valueB"????T   ?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????To
*time_distributed_3/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(time_distributed_3/dropout_1/dropout/MulMul#time_distributed_3/Reshape:output:03time_distributed_3/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????T}
*time_distributed_3/dropout_1/dropout/ShapeShape#time_distributed_3/Reshape:output:0*
T0*
_output_shapes
:?
Atime_distributed_3/dropout_1/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_3/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0x
3time_distributed_3/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
1time_distributed_3/dropout_1/dropout/GreaterEqualGreaterEqualJtime_distributed_3/dropout_1/dropout/random_uniform/RandomUniform:output:0<time_distributed_3/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T?
)time_distributed_3/dropout_1/dropout/CastCast5time_distributed_3/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T?
*time_distributed_3/dropout_1/dropout/Mul_1Mul,time_distributed_3/dropout_1/dropout/Mul:z:0-time_distributed_3/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Tw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
time_distributed_3/Reshape_1Reshape.time_distributed_3/dropout_1/dropout/Mul_1:z:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Ts
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????Tq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
0time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpJtime_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
!time_distributed_4/dense_2/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
1time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpItime_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
"time_distributed_4/dense_2/BiasAddBiasAdd+time_distributed_4/dense_2/MatMul:product:09time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
time_distributed_4/Reshape_1Reshape+time_distributed_4/dense_2/BiasAdd:output:0+time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/Reshape_2Reshape%time_distributed_3/Reshape_1:output:0+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????Tq
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*o
*time_distributed_5/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(time_distributed_5/dropout_2/dropout/MulMul#time_distributed_5/Reshape:output:03time_distributed_5/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????*}
*time_distributed_5/dropout_2/dropout/ShapeShape#time_distributed_5/Reshape:output:0*
T0*
_output_shapes
:?
Atime_distributed_5/dropout_2/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_5/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0x
3time_distributed_5/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
1time_distributed_5/dropout_2/dropout/GreaterEqualGreaterEqualJtime_distributed_5/dropout_2/dropout/random_uniform/RandomUniform:output:0<time_distributed_5/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*?
)time_distributed_5/dropout_2/dropout/CastCast5time_distributed_5/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*?
*time_distributed_5/dropout_2/dropout/Mul_1Mul,time_distributed_5/dropout_2/dropout/Mul:z:0-time_distributed_5/dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*w
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
time_distributed_5/Reshape_1Reshape.time_distributed_5/dropout_2/dropout/Mul_1:z:0+time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
0time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpJtime_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ?
time_distributed_6/Reshape_1Reshape+time_distributed_6/dense_3/BiasAdd:output:0+time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????s
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*s
cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           u
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
:?????????*

begin_mask*
end_maskt
IdentityIdentity!cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_1/MatMul/ReadVariableOp+^lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_2/dense_1/BiasAdd/ReadVariableOp1^time_distributed_2/dense_1/MatMul/ReadVariableOp2^time_distributed_4/dense_2/BiasAdd/ReadVariableOp1^time_distributed_4/dense_2/MatMul/ReadVariableOp2^time_distributed_6/dense_3/BiasAdd/ReadVariableOp1^time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2V
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2T
(lstm_1/lstm_cell_1/MatMul/ReadVariableOp(lstm_1/lstm_cell_1/MatMul/ReadVariableOp2X
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while2^
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
:?????????
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319021

inputs

states
states_1C
/matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
O
;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
3__inference_time_distributed_2_layer_call_fn_324587

inputs,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320832s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_325739
file_prefix0
&assignvariableop_training_adagrad_iter:	 3
)assignvariableop_1_training_adagrad_decay: ;
1assignvariableop_2_training_adagrad_learning_rate: ;
(assignvariableop_3_lstm_lstm_cell_kernel:	?
F
2assignvariableop_4_lstm_lstm_cell_recurrent_kernel:
??
5
&assignvariableop_5_lstm_lstm_cell_bias:	?
@
,assignvariableop_6_lstm_1_lstm_cell_1_kernel:
??
J
6assignvariableop_7_lstm_1_lstm_cell_1_recurrent_kernel:
??
9
*assignvariableop_8_lstm_1_lstm_cell_1_bias:	?
@
,assignvariableop_9_lstm_2_lstm_cell_2_kernel:
??
K
7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernel:
??
:
+assignvariableop_11_lstm_2_lstm_cell_2_bias:	?
?
+assignvariableop_12_time_distributed_kernel:
??8
)assignvariableop_13_time_distributed_bias:	?@
-assignvariableop_14_time_distributed_2_kernel:	?T9
+assignvariableop_15_time_distributed_2_bias:T?
-assignvariableop_16_time_distributed_4_kernel:T*9
+assignvariableop_17_time_distributed_4_bias:*?
-assignvariableop_18_time_distributed_6_kernel:*9
+assignvariableop_19_time_distributed_6_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: Y
Fassignvariableop_24_training_adagrad_lstm_lstm_cell_kernel_accumulator:	?
d
Passignvariableop_25_training_adagrad_lstm_lstm_cell_recurrent_kernel_accumulator:
??
S
Dassignvariableop_26_training_adagrad_lstm_lstm_cell_bias_accumulator:	?
^
Jassignvariableop_27_training_adagrad_lstm_1_lstm_cell_1_kernel_accumulator:
??
h
Tassignvariableop_28_training_adagrad_lstm_1_lstm_cell_1_recurrent_kernel_accumulator:
??
W
Hassignvariableop_29_training_adagrad_lstm_1_lstm_cell_1_bias_accumulator:	?
^
Jassignvariableop_30_training_adagrad_lstm_2_lstm_cell_2_kernel_accumulator:
??
h
Tassignvariableop_31_training_adagrad_lstm_2_lstm_cell_2_recurrent_kernel_accumulator:
??
W
Hassignvariableop_32_training_adagrad_lstm_2_lstm_cell_2_bias_accumulator:	?
\
Hassignvariableop_33_training_adagrad_time_distributed_kernel_accumulator:
??U
Fassignvariableop_34_training_adagrad_time_distributed_bias_accumulator:	?]
Jassignvariableop_35_training_adagrad_time_distributed_2_kernel_accumulator:	?TV
Hassignvariableop_36_training_adagrad_time_distributed_2_bias_accumulator:T\
Jassignvariableop_37_training_adagrad_time_distributed_4_kernel_accumulator:T*V
Hassignvariableop_38_training_adagrad_time_distributed_4_bias_accumulator:*\
Jassignvariableop_39_training_adagrad_time_distributed_6_kernel_accumulator:*V
Hassignvariableop_40_training_adagrad_time_distributed_6_bias_accumulator:
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/12/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/13/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/14/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/15/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/16/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_training_adagrad_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_training_adagrad_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp1assignvariableop_2_training_adagrad_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_lstm_lstm_cell_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_lstm_lstm_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_lstm_lstm_cell_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_lstm_1_lstm_cell_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_lstm_1_lstm_cell_1_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_lstm_1_lstm_cell_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_2_lstm_cell_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_2_lstm_cell_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_time_distributed_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_time_distributed_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_time_distributed_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_time_distributed_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_time_distributed_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_time_distributed_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_time_distributed_6_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_time_distributed_6_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpFassignvariableop_24_training_adagrad_lstm_lstm_cell_kernel_accumulatorIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpPassignvariableop_25_training_adagrad_lstm_lstm_cell_recurrent_kernel_accumulatorIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpDassignvariableop_26_training_adagrad_lstm_lstm_cell_bias_accumulatorIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpJassignvariableop_27_training_adagrad_lstm_1_lstm_cell_1_kernel_accumulatorIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpTassignvariableop_28_training_adagrad_lstm_1_lstm_cell_1_recurrent_kernel_accumulatorIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpHassignvariableop_29_training_adagrad_lstm_1_lstm_cell_1_bias_accumulatorIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpJassignvariableop_30_training_adagrad_lstm_2_lstm_cell_2_kernel_accumulatorIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpTassignvariableop_31_training_adagrad_lstm_2_lstm_cell_2_recurrent_kernel_accumulatorIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpHassignvariableop_32_training_adagrad_lstm_2_lstm_cell_2_bias_accumulatorIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpHassignvariableop_33_training_adagrad_time_distributed_kernel_accumulatorIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpFassignvariableop_34_training_adagrad_time_distributed_bias_accumulatorIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpJassignvariableop_35_training_adagrad_time_distributed_2_kernel_accumulatorIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpHassignvariableop_36_training_adagrad_time_distributed_2_bias_accumulatorIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpJassignvariableop_37_training_adagrad_time_distributed_4_kernel_accumulatorIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpHassignvariableop_38_training_adagrad_time_distributed_4_bias_accumulatorIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpJassignvariableop_39_training_adagrad_time_distributed_6_kernel_accumulatorIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpHassignvariableop_40_training_adagrad_time_distributed_6_bias_accumulatorIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
?
F
*__inference_dropout_2_layer_call_fn_325421

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
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_319978`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????*"
identityIdentity:output:0*&
_input_shapes
:?????????*:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?1
?
while_body_323429
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__wrapped_model_318634

lstm_inputX
Esequential_lstm_lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
e
Qsequential_lstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
S
Dsequential_lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
a
Msequential_lstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
m
Ysequential_lstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
[
Lsequential_lstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
a
Msequential_lstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
m
Ysequential_lstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
[
Lsequential_lstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
c
Osequential_time_distributed_dense_matmul_readvariableop_time_distributed_kernel:
??]
Nsequential_time_distributed_dense_biasadd_readvariableop_time_distributed_bias:	?h
Usequential_time_distributed_2_dense_1_matmul_readvariableop_time_distributed_2_kernel:	?Tb
Tsequential_time_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias:Tg
Usequential_time_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel:T*b
Tsequential_time_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias:*g
Usequential_time_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel:*b
Tsequential_time_distributed_6_dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp?/sequential/lstm/lstm_cell/MatMul/ReadVariableOp?1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp?sequential/lstm/while?4sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp?3sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOp?5sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp?sequential/lstm_1/while?4sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?3sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?5sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?sequential/lstm_2/while?8sequential/time_distributed/dense/BiasAdd/ReadVariableOp?7sequential/time_distributed/dense/MatMul/ReadVariableOp?<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp?;sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp?<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp?;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp?<sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp?;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOpO
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
B :??
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
:??????????c
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
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
:??????????s
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose	Transpose
lstm_input'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:?????????d
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
:	?
*
dtype0?
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:07sequential/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpQsequential_lstm_lstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
"sequential/lstm/lstm_cell/MatMul_1MatMulsequential/lstm/zeros:output:09sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/MatMul:product:0,sequential/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDsequential_lstm_lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
!sequential/lstm/lstm_cell/BiasAddBiasAdd!sequential/lstm/lstm_cell/add:z:08sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
k
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:0*sequential/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
!sequential/lstm/lstm_cell/SigmoidSigmoid(sequential/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid(sequential/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mulMul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mul_1Mul%sequential/lstm/lstm_cell/Sigmoid:y:0(sequential/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/add_1AddV2!sequential/lstm/lstm_cell/mul:z:0#sequential/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid(sequential/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell/mul_2Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0#sequential/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!sequential_lstm_while_body_318209*-
cond%R#
!sequential_lstm_while_cond_318208*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????k
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    f
sequential/lstm_1/ShapeShapesequential/lstm/transpose_1:y:0*
T0*
_output_shapes
:o
%sequential/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm_1/strided_sliceStridedSlice sequential/lstm_1/Shape:output:0.sequential/lstm_1/strided_slice/stack:output:00sequential/lstm_1/strided_slice/stack_1:output:00sequential/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm_1/zeros/packedPack(sequential/lstm_1/strided_slice:output:0)sequential/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm_1/zerosFill'sequential/lstm_1/zeros/packed:output:0&sequential/lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????e
"sequential/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
 sequential/lstm_1/zeros_1/packedPack(sequential/lstm_1/strided_slice:output:0+sequential/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm_1/zeros_1Fill)sequential/lstm_1/zeros_1/packed:output:0(sequential/lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????u
 sequential/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm_1/transpose	Transposesequential/lstm/transpose_1:y:0)sequential/lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????h
sequential/lstm_1/Shape_1Shapesequential/lstm_1/transpose:y:0*
T0*
_output_shapes
:q
'sequential/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_1/strided_slice_1StridedSlice"sequential/lstm_1/Shape_1:output:00sequential/lstm_1/strided_slice_1/stack:output:02sequential/lstm_1/strided_slice_1/stack_1:output:02sequential/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-sequential/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential/lstm_1/TensorArrayV2TensorListReserve6sequential/lstm_1/TensorArrayV2/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Gsequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
9sequential/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm_1/transpose:y:0Psequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???q
'sequential/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_1/strided_slice_2StridedSlicesequential/lstm_1/transpose:y:00sequential/lstm_1/strided_slice_2/stack:output:02sequential/lstm_1/strided_slice_2/stack_1:output:02sequential/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
3sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpMsequential_lstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
$sequential/lstm_1/lstm_cell_1/MatMulMatMul*sequential/lstm_1/strided_slice_2:output:0;sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
5sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpYsequential_lstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
&sequential/lstm_1/lstm_cell_1/MatMul_1MatMul sequential/lstm_1/zeros:output:0=sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!sequential/lstm_1/lstm_cell_1/addAddV2.sequential/lstm_1/lstm_cell_1/MatMul:product:00sequential/lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
4sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_lstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
%sequential/lstm_1/lstm_cell_1/BiasAddBiasAdd%sequential/lstm_1/lstm_cell_1/add:z:0<sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
o
-sequential/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential/lstm_1/lstm_cell_1/splitSplit6sequential/lstm_1/lstm_cell_1/split/split_dim:output:0.sequential/lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
%sequential/lstm_1/lstm_cell_1/SigmoidSigmoid,sequential/lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid,sequential/lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:???????????
!sequential/lstm_1/lstm_cell_1/mulMul+sequential/lstm_1/lstm_cell_1/Sigmoid_1:y:0"sequential/lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#sequential/lstm_1/lstm_cell_1/mul_1Mul)sequential/lstm_1/lstm_cell_1/Sigmoid:y:0,sequential/lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:???????????
#sequential/lstm_1/lstm_cell_1/add_1AddV2%sequential/lstm_1/lstm_cell_1/mul:z:0'sequential/lstm_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:???????????
'sequential/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid,sequential/lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:???????????
#sequential/lstm_1/lstm_cell_1/mul_2Mul+sequential/lstm_1/lstm_cell_1/Sigmoid_2:y:0'sequential/lstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
/sequential/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
!sequential/lstm_1/TensorArrayV2_1TensorListReserve8sequential/lstm_1/TensorArrayV2_1/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
sequential/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*sequential/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????f
$sequential/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/lstm_1/whileWhile-sequential/lstm_1/while/loop_counter:output:03sequential/lstm_1/while/maximum_iterations:output:0sequential/lstm_1/time:output:0*sequential/lstm_1/TensorArrayV2_1:handle:0 sequential/lstm_1/zeros:output:0"sequential/lstm_1/zeros_1:output:0*sequential/lstm_1/strided_slice_1:output:0Isequential/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_lstm_1_lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelYsequential_lstm_1_lstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelLsequential_lstm_1_lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#sequential_lstm_1_while_body_318344*/
cond'R%
#sequential_lstm_1_while_cond_318343*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
Bsequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
4sequential/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack sequential/lstm_1/while:output:3Ksequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0z
'sequential/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????s
)sequential/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_1/strided_slice_3StridedSlice=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:00sequential/lstm_1/strided_slice_3/stack:output:02sequential/lstm_1/strided_slice_3/stack_1:output:02sequential/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskw
"sequential/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm_1/transpose_1	Transpose=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0+sequential/lstm_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????m
sequential/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
sequential/lstm_2/ShapeShape!sequential/lstm_1/transpose_1:y:0*
T0*
_output_shapes
:o
%sequential/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm_2/strided_sliceStridedSlice sequential/lstm_2/Shape:output:0.sequential/lstm_2/strided_slice/stack:output:00sequential/lstm_2/strided_slice/stack_1:output:00sequential/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm_2/zeros/packedPack(sequential/lstm_2/strided_slice:output:0)sequential/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm_2/zerosFill'sequential/lstm_2/zeros/packed:output:0&sequential/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????e
"sequential/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
 sequential/lstm_2/zeros_1/packedPack(sequential/lstm_2/strided_slice:output:0+sequential/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm_2/zeros_1Fill)sequential/lstm_2/zeros_1/packed:output:0(sequential/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????u
 sequential/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm_2/transpose	Transpose!sequential/lstm_1/transpose_1:y:0)sequential/lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????h
sequential/lstm_2/Shape_1Shapesequential/lstm_2/transpose:y:0*
T0*
_output_shapes
:q
'sequential/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_2/strided_slice_1StridedSlice"sequential/lstm_2/Shape_1:output:00sequential/lstm_2/strided_slice_1/stack:output:02sequential/lstm_2/strided_slice_1/stack_1:output:02sequential/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-sequential/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential/lstm_2/TensorArrayV2TensorListReserve6sequential/lstm_2/TensorArrayV2/element_shape:output:0*sequential/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Gsequential/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
9sequential/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm_2/transpose:y:0Psequential/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???q
'sequential/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_2/strided_slice_2StridedSlicesequential/lstm_2/transpose:y:00sequential/lstm_2/strided_slice_2/stack:output:02sequential/lstm_2/strided_slice_2/stack_1:output:02sequential/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
3sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpMsequential_lstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
$sequential/lstm_2/lstm_cell_2/MatMulMatMul*sequential/lstm_2/strided_slice_2:output:0;sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
5sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpYsequential_lstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
&sequential/lstm_2/lstm_cell_2/MatMul_1MatMul sequential/lstm_2/zeros:output:0=sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!sequential/lstm_2/lstm_cell_2/addAddV2.sequential/lstm_2/lstm_cell_2/MatMul:product:00sequential/lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
4sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpLsequential_lstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
%sequential/lstm_2/lstm_cell_2/BiasAddBiasAdd%sequential/lstm_2/lstm_cell_2/add:z:0<sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
o
-sequential/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential/lstm_2/lstm_cell_2/splitSplit6sequential/lstm_2/lstm_cell_2/split/split_dim:output:0.sequential/lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
%sequential/lstm_2/lstm_cell_2/SigmoidSigmoid,sequential/lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm_2/lstm_cell_2/Sigmoid_1Sigmoid,sequential/lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
!sequential/lstm_2/lstm_cell_2/mulMul+sequential/lstm_2/lstm_cell_2/Sigmoid_1:y:0"sequential/lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#sequential/lstm_2/lstm_cell_2/mul_1Mul)sequential/lstm_2/lstm_cell_2/Sigmoid:y:0,sequential/lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
#sequential/lstm_2/lstm_cell_2/add_1AddV2%sequential/lstm_2/lstm_cell_2/mul:z:0'sequential/lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
'sequential/lstm_2/lstm_cell_2/Sigmoid_2Sigmoid,sequential/lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
#sequential/lstm_2/lstm_cell_2/mul_2Mul+sequential/lstm_2/lstm_cell_2/Sigmoid_2:y:0'sequential/lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
/sequential/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
!sequential/lstm_2/TensorArrayV2_1TensorListReserve8sequential/lstm_2/TensorArrayV2_1/element_shape:output:0*sequential/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
sequential/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*sequential/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????f
$sequential/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/lstm_2/whileWhile-sequential/lstm_2/while/loop_counter:output:03sequential/lstm_2/while/maximum_iterations:output:0sequential/lstm_2/time:output:0*sequential/lstm_2/TensorArrayV2_1:handle:0 sequential/lstm_2/zeros:output:0"sequential/lstm_2/zeros_1:output:0*sequential/lstm_2/strided_slice_1:output:0Isequential/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_lstm_2_lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelYsequential_lstm_2_lstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelLsequential_lstm_2_lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#sequential_lstm_2_while_body_318479*/
cond'R%
#sequential_lstm_2_while_cond_318478*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
Bsequential/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
4sequential/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack sequential/lstm_2/while:output:3Ksequential/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0z
'sequential/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????s
)sequential/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)sequential/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/lstm_2/strided_slice_3StridedSlice=sequential/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:00sequential/lstm_2/strided_slice_3/stack:output:02sequential/lstm_2/strided_slice_3/stack_1:output:02sequential/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskw
"sequential/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm_2/transpose_1	Transpose=sequential/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0+sequential/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????m
sequential/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    z
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#sequential/time_distributed/ReshapeReshape!sequential/lstm_2/transpose_1:y:02sequential/time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
7sequential/time_distributed/dense/MatMul/ReadVariableOpReadVariableOpOsequential_time_distributed_dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
+sequential/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   ?
%sequential/time_distributed/Reshape_1Reshape2sequential/time_distributed/dense/BiasAdd:output:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????|
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
%sequential/time_distributed/Reshape_2Reshape!sequential/lstm_2/transpose_1:y:04sequential/time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:??????????|
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
valueB"????   ?   ?
'sequential/time_distributed_1/Reshape_1Reshape7sequential/time_distributed_1/dropout/Identity:output:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????~
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
:	?T*
dtype0?
,sequential/time_distributed_2/dense_1/MatMulMatMul.sequential/time_distributed_2/Reshape:output:0Csequential/time_distributed_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
<sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpTsequential_time_distributed_2_dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
-sequential/time_distributed_2/dense_1/BiasAddBiasAdd6sequential/time_distributed_2/dense_1/MatMul:product:0Dsequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
-sequential/time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
'sequential/time_distributed_2/Reshape_1Reshape6sequential/time_distributed_2/dense_1/BiasAdd:output:06sequential/time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T~
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
valueB"????T   ?
%sequential/time_distributed_3/ReshapeReshape0sequential/time_distributed_2/Reshape_1:output:04sequential/time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
0sequential/time_distributed_3/dropout_1/IdentityIdentity.sequential/time_distributed_3/Reshape:output:0*
T0*'
_output_shapes
:?????????T?
-sequential/time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
'sequential/time_distributed_3/Reshape_1Reshape9sequential/time_distributed_3/dropout_1/Identity:output:06sequential/time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T~
-sequential/time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
'sequential/time_distributed_3/Reshape_2Reshape0sequential/time_distributed_2/Reshape_1:output:06sequential/time_distributed_3/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????T|
+sequential/time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
%sequential/time_distributed_4/ReshapeReshape0sequential/time_distributed_3/Reshape_1:output:04sequential/time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
;sequential/time_distributed_4/dense_2/MatMul/ReadVariableOpReadVariableOpUsequential_time_distributed_4_dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
,sequential/time_distributed_4/dense_2/MatMulMatMul.sequential/time_distributed_4/Reshape:output:0Csequential/time_distributed_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
<sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOpReadVariableOpTsequential_time_distributed_4_dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
-sequential/time_distributed_4/dense_2/BiasAddBiasAdd6sequential/time_distributed_4/dense_2/MatMul:product:0Dsequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
-sequential/time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
'sequential/time_distributed_4/Reshape_1Reshape6sequential/time_distributed_4/dense_2/BiasAdd:output:06sequential/time_distributed_4/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*~
-sequential/time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
'sequential/time_distributed_4/Reshape_2Reshape0sequential/time_distributed_3/Reshape_1:output:06sequential/time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????T|
+sequential/time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
%sequential/time_distributed_5/ReshapeReshape0sequential/time_distributed_4/Reshape_1:output:04sequential/time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
0sequential/time_distributed_5/dropout_2/IdentityIdentity.sequential/time_distributed_5/Reshape:output:0*
T0*'
_output_shapes
:?????????*?
-sequential/time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
'sequential/time_distributed_5/Reshape_1Reshape9sequential/time_distributed_5/dropout_2/Identity:output:06sequential/time_distributed_5/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*~
-sequential/time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
'sequential/time_distributed_5/Reshape_2Reshape0sequential/time_distributed_4/Reshape_1:output:06sequential/time_distributed_5/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*|
+sequential/time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
%sequential/time_distributed_6/ReshapeReshape0sequential/time_distributed_5/Reshape_1:output:04sequential/time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
;sequential/time_distributed_6/dense_3/MatMul/ReadVariableOpReadVariableOpUsequential_time_distributed_6_dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ?
'sequential/time_distributed_6/Reshape_1Reshape6sequential/time_distributed_6/dense_3/BiasAdd:output:06sequential/time_distributed_6/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????~
-sequential/time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
'sequential/time_distributed_6/Reshape_2Reshape0sequential/time_distributed_5/Reshape_1:output:06sequential/time_distributed_6/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????*~
)sequential/cropping1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           ?
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
:?????????*

begin_mask*
end_mask
IdentityIdentity,sequential/cropping1d/strided_slice:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp1^sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp0^sequential/lstm/lstm_cell/MatMul/ReadVariableOp2^sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp^sequential/lstm/while5^sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp4^sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOp6^sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^sequential/lstm_1/while5^sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp4^sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOp6^sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^sequential/lstm_2/while9^sequential/time_distributed/dense/BiasAdd/ReadVariableOp8^sequential/time_distributed/dense/MatMul/ReadVariableOp=^sequential/time_distributed_2/dense_1/BiasAdd/ReadVariableOp<^sequential/time_distributed_2/dense_1/MatMul/ReadVariableOp=^sequential/time_distributed_4/dense_2/BiasAdd/ReadVariableOp<^sequential/time_distributed_4/dense_2/MatMul/ReadVariableOp=^sequential/time_distributed_6/dense_3/BiasAdd/ReadVariableOp<^sequential/time_distributed_6/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2d
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp2b
/sequential/lstm/lstm_cell/MatMul/ReadVariableOp/sequential/lstm/lstm_cell/MatMul/ReadVariableOp2f
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while2l
4sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp4sequential/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2j
3sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOp3sequential/lstm_1/lstm_cell_1/MatMul/ReadVariableOp2n
5sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp5sequential/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp22
sequential/lstm_1/whilesequential/lstm_1/while2l
4sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp4sequential/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2j
3sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOp3sequential/lstm_2/lstm_cell_2/MatMul/ReadVariableOp2n
5sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp5sequential/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp22
sequential/lstm_2/whilesequential/lstm_2/while2t
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
:?????????
$
_user_specified_name
lstm_input
?
?
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319343

inputs

states
states_1C
/matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
O
;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
,__inference_lstm_cell_2_layer_call_fn_325251

inputs
states_0
states_1-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319475p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
3__inference_time_distributed_2_layer_call_fn_324580

inputs,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320598s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_320180
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_320180___redundant_placeholder0.
*while_cond_320180___redundant_placeholder1.
*while_cond_320180___redundant_placeholder2.
*while_cond_320180___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_320008

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????*C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????*"
identityIdentity:output:0*&
_input_shapes
:?????????*:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?1
?
while_body_320467
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_319353
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_319353___redundant_placeholder0.
*while_cond_319353___redundant_placeholder1.
*while_cond_319353___redundant_placeholder2.
*while_cond_319353___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?0
?
lstm_while_body_321685
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_325039

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
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
?
?
,__inference_lstm_cell_1_layer_call_fn_325149

inputs
states_0
states_1-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319021p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325193

inputs
states_0
states_1C
/matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
O
;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_322701
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_322701___redundant_placeholder0.
*while_cond_322701___redundant_placeholder1.
*while_cond_322701___redundant_placeholder2.
*while_cond_322701___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
? 
?
while_body_319528
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
'lstm_cell_2_lstm_2_lstm_cell_2_kernel_0:
??
E
1lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
4
%lstm_cell_2_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
%lstm_cell_2_lstm_2_lstm_cell_2_kernel:
??
C
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
2
#lstm_cell_2_lstm_2_lstm_cell_2_bias:	?
??#lstm_cell_2/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'lstm_cell_2_lstm_2_lstm_cell_2_kernel_01lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0%lstm_cell_2_lstm_2_lstm_cell_2_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319475?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_2/StatefulPartitionedCall:output:0*
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
: ~

Identity_4Identity,lstm_cell_2/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_5Identity,lstm_cell_2/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"L
#lstm_cell_2_lstm_2_lstm_cell_2_bias%lstm_cell_2_lstm_2_lstm_cell_2_bias_0"P
%lstm_cell_2_lstm_2_lstm_cell_2_kernel'lstm_cell_2_lstm_2_lstm_cell_2_kernel_0"d
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel1lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_323960
inputs_0O
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while=
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323878*
condR
while_cond_323877*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
+__inference_sequential_layer_call_fn_321628

inputs(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?
-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?
-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?
+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*+
time_distributed_6_kernel:*%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biaslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_biaslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_321426s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
while_body_323568
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324426

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
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
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
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
%:???????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
'__inference_lstm_1_layer_call_fn_323233

inputs-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_321195t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_3_layer_call_fn_324672

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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320612d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????T"
identityIdentity:output:0**
_input_shapes
:?????????T:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
'__inference_lstm_2_layer_call_fn_323797
inputs_0-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_319420}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
b
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
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
:?????????*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_319985

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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dropout_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_319978\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape"dropout_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????*"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_325443

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????*C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????*"
identityIdentity:output:0*&
_input_shapes
:?????????*:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?2
?
lstm_1_while_body_322302
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
value	B :\
add_1AddV2lstm_1_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: _

Identity_1Identitylstm_1_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"?
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_323707
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
l
3__inference_time_distributed_5_layer_call_fn_324849

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
 :??????????????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320029|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_320056

inputsA
/matmul_readvariableop_time_distributed_6_kernel:*<
.biasadd_readvariableop_time_distributed_6_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
:?????????*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
while_cond_318709
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_318709___redundant_placeholder0.
*while_cond_318709___redundant_placeholder1.
*while_cond_318709___redundant_placeholder2.
*while_cond_318709___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_321350

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_321268*
condR
while_cond_321267*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319475

inputs

states
states_1C
/matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
O
;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?2
?
lstm_1_while_body_321820
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
value	B :\
add_1AddV2lstm_1_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: _

Identity_1Identitylstm_1_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"?
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
l
3__inference_time_distributed_5_layer_call_fn_324859

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
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320752s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????*22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?0
?
while_body_322841
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320629

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ~
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
while_body_318884
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!lstm_cell_lstm_lstm_cell_kernel_0:	?
?
+lstm_cell_lstm_lstm_cell_recurrent_kernel_0:
??
.
lstm_cell_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
lstm_cell_lstm_lstm_cell_kernel:	?
=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??
,
lstm_cell_lstm_lstm_cell_bias:	?
??!lstm_cell/StatefulPartitionedCall?
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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318831?
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
:??????????|

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????j
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
:: : : : :??????????:??????????: : : : : 2F
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?O
?
F__inference_sequential_layer_call_and_return_conditional_losses_321560

lstm_input-
lstm_lstm_lstm_cell_kernel:	?
8
$lstm_lstm_lstm_cell_recurrent_kernel:
??
'
lstm_lstm_lstm_cell_bias:	?
4
 lstm_1_lstm_1_lstm_cell_1_kernel:
??
>
*lstm_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
-
lstm_1_lstm_1_lstm_cell_1_bias:	?
4
 lstm_2_lstm_2_lstm_cell_2_kernel:
??
>
*lstm_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
-
lstm_2_lstm_2_lstm_cell_2_bias:	?
<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?T8
*time_distributed_2_time_distributed_2_bias:T>
,time_distributed_4_time_distributed_4_kernel:T*8
*time_distributed_4_time_distributed_4_bias:*>
,time_distributed_6_time_distributed_6_kernel:*8
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_3/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_5/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_321350?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 lstm_1_lstm_1_lstm_cell_1_kernel*lstm_1_lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_321195?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 lstm_2_lstm_2_lstm_cell_2_kernel*lstm_2_lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_321040?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320886o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshape'lstm_2/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320860q
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320832q
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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320806q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_3/StatefulPartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320778q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape3time_distributed_3/StatefulPartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_5/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0+^time_distributed_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320752q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_5/StatefulPartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320724q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape3time_distributed_5/StatefulPartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_5/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_5/StatefulPartitionedCall*time_distributed_5/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
?
?
3__inference_time_distributed_4_layer_call_fn_324748

inputs+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_319921|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320724

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325021

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?3
?	
#sequential_lstm_1_while_body_318344(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#sequential_lstm_1_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0:
??
]
Ilstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
K
<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!sequential_lstm_1_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?
??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulplaceholder_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_1/mul_2:z:0*
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
value	B :g
add_1AddV2$sequential_lstm_1_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: j

Identity_1Identity*sequential_lstm_1_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_1/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_1/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias<lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias_0"?
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernelIlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel_0"|
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel=lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel_0"H
!sequential_lstm_1_strided_slice_1#sequential_lstm_1_strided_slice_1_0"?
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_319624

inputsA
-matmul_readvariableop_time_distributed_kernel:
??;
,biasadd_readvariableop_time_distributed_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324543

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
valueB"????   ?   ?
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_322645

inputs(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_321350t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_319690

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
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_325343

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
?
G
+__inference_cropping1d_layer_call_fn_325031

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324099
inputs_0O
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while=
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_324017*
condR
while_cond_324016*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_319720

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
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
 *??L>?
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
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325105

inputs
states_0
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?
K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_321112
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_321112___redundant_placeholder0.
*while_cond_321112___redundant_placeholder1.
*while_cond_321112___redundant_placeholder2.
*while_cond_321112___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_319741

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
C__inference_dropout_layer_call_and_return_conditional_losses_319720\
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
?
?
3__inference_time_distributed_4_layer_call_fn_324769

inputs+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320778s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_4_layer_call_fn_324762

inputs+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_kerneltime_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320629s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?8
?
@__inference_lstm_layer_call_and_return_conditional_losses_318776

inputs2
lstm_cell_lstm_lstm_cell_kernel:	?
=
)lstm_cell_lstm_lstm_cell_recurrent_kernel:
??
,
lstm_cell_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318699n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_318710*
condR
while_cond_318709*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r
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
?
,__inference_lstm_cell_2_layer_call_fn_325237

inputs
states_0
states_1-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1lstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319343p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324447

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
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
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
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
%:???????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324534

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
 *  ??
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
 *??L>?
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
?I
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323650

inputsO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323568*
condR
while_cond_323567*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324643

inputsJ
7dense_1_matmul_readvariableop_time_distributed_2_kernel:	?TD
6dense_1_biasadd_readvariableop_time_distributed_2_bias:T
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
:	?T*
dtype0?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ~
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????Te
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????T?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_cropping1d_layer_call_fn_325026

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
F__inference_cropping1d_layer_call_and_return_conditional_losses_320115v
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
?
m
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320806

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMulReshape:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????TW
dropout_1/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Td
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   T   ?
	Reshape_1Reshapedropout_1/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????T^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????T"
identityIdentity:output:0**
_input_shapes
:?????????T:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?1
?
while_body_323878
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?I
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_320406

inputsO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320324*
condR
while_cond_320323*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_325355

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
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
 *??L>?
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
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320752

inputs
identity?^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????*W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*^
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:?????????*"
identityIdentity:output:0**
_input_shapes
:?????????*:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_5_layer_call_fn_324844

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
 :??????????????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_319985m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????*"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_320549

inputsO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_320467*
condR
while_cond_320466*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_323062

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322980*
condR
while_cond_322979*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324898

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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMulReshape:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????*W
dropout_2/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????**
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????*?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????*?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????*\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_2/dropout/Mul_1:z:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????*"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_325416

inputsA
/matmul_readvariableop_time_distributed_4_kernel:T*<
.biasadd_readvariableop_time_distributed_4_bias:*
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????*w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324839

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ~
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?I
?

F__inference_sequential_layer_call_and_return_conditional_losses_320676

inputs-
lstm_lstm_lstm_cell_kernel:	?
8
$lstm_lstm_lstm_cell_recurrent_kernel:
??
'
lstm_lstm_lstm_cell_bias:	?
4
 lstm_1_lstm_1_lstm_cell_1_kernel:
??
>
*lstm_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
-
lstm_1_lstm_1_lstm_cell_1_bias:	?
4
 lstm_2_lstm_2_lstm_cell_2_kernel:
??
>
*lstm_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
-
lstm_2_lstm_2_lstm_cell_2_bias:	?
<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?T8
*time_distributed_2_time_distributed_2_bias:T>
,time_distributed_4_time_distributed_4_kernel:T*8
*time_distributed_4_time_distributed_4_bias:*>
,time_distributed_6_time_distributed_6_kernel:*8
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_320263?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 lstm_1_lstm_1_lstm_cell_1_kernel*lstm_1_lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_320406?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 lstm_2_lstm_2_lstm_cell_2_kernel*lstm_2_lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_320549?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320567o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshape'lstm_2/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320581q
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320598q
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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320612q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320629q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape+time_distributed_3/PartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320643q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320660q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_324930

inputs+
time_distributed_6_kernel:*%
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
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320065|
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
$:??????????????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_319954

inputs3
!dense_2_time_distributed_4_kernel:T*-
dense_2_time_distributed_4_bias:*
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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0!dense_2_time_distributed_4_kerneldense_2_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_319912\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
?
lstm_1_while_cond_321819
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_15
1lstm_1_while_cond_321819___redundant_placeholder05
1lstm_1_while_cond_321819___redundant_placeholder15
1lstm_1_while_cond_321819___redundant_placeholder25
1lstm_1_while_cond_321819___redundant_placeholder3
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
l
3__inference_time_distributed_3_layer_call_fn_324677

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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320806s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????T22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
while_cond_323289
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323289___redundant_placeholder0.
*while_cond_323289___redundant_placeholder1.
*while_cond_323289___redundant_placeholder2.
*while_cond_323289___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_324155
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_324155___redundant_placeholder0.
*while_cond_324155___redundant_placeholder1.
*while_cond_324155___redundant_placeholder2.
*while_cond_324155___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320660

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
6dense_3_biasadd_readvariableop_time_distributed_6_bias:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
valueB"????      ~
	Reshape_1Reshapedense_3/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_319810

inputs4
!dense_1_time_distributed_2_kernel:	?T-
dense_1_time_distributed_2_bias:T
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_319768\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????Th
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
?
c
*__inference_dropout_1_layer_call_fn_325382

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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_319864o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*&
_input_shapes
:?????????T22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_6_layer_call_fn_324951

inputs+
time_distributed_6_kernel:*%
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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320724s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
'__inference_lstm_1_layer_call_fn_323225

inputs-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_320406t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_lstm_2_layer_call_fn_323805
inputs_0-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0lstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_319594}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?1
?
while_body_324017
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
1__inference_time_distributed_layer_call_fn_324391

inputs+
time_distributed_kernel:
??$
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_319666}
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
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_322840
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_322840___redundant_placeholder0.
*while_cond_322840___redundant_placeholder1.
*while_cond_322840___redundant_placeholder2.
*while_cond_322840___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_319527
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_319527___redundant_placeholder0.
*while_cond_319527___redundant_placeholder1.
*while_cond_319527___redundant_placeholder2.
*while_cond_319527___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
3__inference_time_distributed_2_layer_call_fn_324573

inputs,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_2_kerneltime_distributed_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_319810|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????T`
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
(__inference_dropout_layer_call_fn_325333

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
C__inference_dropout_layer_call_and_return_conditional_losses_319690a
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
?I
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324238

inputsO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?

identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernelGlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_324156*
condR
while_cond_324155*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
lstm_while_cond_322166
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_13
/lstm_while_cond_322166___redundant_placeholder03
/lstm_while_cond_322166___redundant_placeholder13
/lstm_while_cond_322166___redundant_placeholder23
/lstm_while_cond_322166___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_323118
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_323118___redundant_placeholder0.
*while_cond_323118___redundant_placeholder1.
*while_cond_323118___redundant_placeholder2.
*while_cond_323118___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
@__inference_lstm_layer_call_and_return_conditional_losses_323201

inputsH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?

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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
:	?
*
dtype0?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAlstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????r
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323119*
condR
while_cond_323118*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*0
_input_shapes
:?????????: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324461

inputsG
3dense_matmul_readvariableop_time_distributed_kernel:
??A
2dense_biasadd_readvariableop_time_distributed_bias:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_time_distributed_kernel* 
_output_shapes
:
??*
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
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   ?   }
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityReshape_1:output:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_325426

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
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_320008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*&
_input_shapes
:?????????*22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
1__inference_time_distributed_layer_call_fn_324398

inputs+
time_distributed_kernel:
??$
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320567t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_318699

inputs

states
states_1>
+matmul_readvariableop_lstm_lstm_cell_kernel:	?
K
7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
9
*biasadd_readvariableop_lstm_lstm_cell_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_lstm_lstm_cell_kernel*
_output_shapes
:	?
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel* 
_output_shapes
:
??
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
~
BiasAdd/ReadVariableOpReadVariableOp*biasadd_readvariableop_lstm_lstm_cell_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?1
?
!sequential_lstm_while_body_318209&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!sequential_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0J
7lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel_0:	?
W
Clstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0:
??
E
6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5#
sequential_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorH
5lstm_cell_matmul_readvariableop_lstm_lstm_cell_kernel:	?
U
Alstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel:
??
C
4lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias:	?
?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?
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
:	?
*
dtype0?
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOpClstm_cell_matmul_1_readvariableop_lstm_lstm_cell_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell/MatMul_1MatMulplaceholder_2)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_cell_biasadd_readvariableop_lstm_lstm_cell_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:??????????o
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????z
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????s
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:??????????k
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:??????????w
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
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
:??????????e

Identity_5Identitylstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
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
:: : : : :??????????:??????????: : : : : 2D
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_319885

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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_319864\
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
value	B :T?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape*dropout_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????Tn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????Tj
NoOpNoOp"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*3
_input_shapes"
 :??????????????????T2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?O
?
F__inference_sequential_layer_call_and_return_conditional_losses_321426

inputs-
lstm_lstm_lstm_cell_kernel:	?
8
$lstm_lstm_lstm_cell_recurrent_kernel:
??
'
lstm_lstm_lstm_cell_bias:	?
4
 lstm_1_lstm_1_lstm_cell_1_kernel:
??
>
*lstm_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
-
lstm_1_lstm_1_lstm_cell_1_bias:	?
4
 lstm_2_lstm_2_lstm_cell_2_kernel:
??
>
*lstm_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
-
lstm_2_lstm_2_lstm_cell_2_bias:	?
<
(time_distributed_time_distributed_kernel:
??5
&time_distributed_time_distributed_bias:	??
,time_distributed_2_time_distributed_2_kernel:	?T8
*time_distributed_2_time_distributed_2_bias:T>
,time_distributed_4_time_distributed_4_kernel:T*8
*time_distributed_4_time_distributed_4_bias:*>
,time_distributed_6_time_distributed_6_kernel:*8
*time_distributed_6_time_distributed_6_bias:
identity??lstm/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?*time_distributed_2/StatefulPartitionedCall?*time_distributed_3/StatefulPartitionedCall?*time_distributed_4/StatefulPartitionedCall?*time_distributed_5/StatefulPartitionedCall?*time_distributed_6/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_lstm_cell_kernel$lstm_lstm_lstm_cell_recurrent_kernellstm_lstm_lstm_cell_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_321350?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 lstm_1_lstm_1_lstm_cell_1_kernel*lstm_1_lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_1_lstm_cell_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_321195?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 lstm_2_lstm_2_lstm_cell_2_kernel*lstm_2_lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_2_lstm_cell_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_321040?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0(time_distributed_time_distributed_kernel&time_distributed_time_distributed_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_320886o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
time_distributed/ReshapeReshape'lstm_2/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320860q
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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_320832q
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
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_320806q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_3/ReshapeReshape3time_distributed_2/StatefulPartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_3/StatefulPartitionedCall:output:0,time_distributed_4_time_distributed_4_kernel*time_distributed_4_time_distributed_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_320778q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   ?
time_distributed_4/ReshapeReshape3time_distributed_3/StatefulPartitionedCall:output:0)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
*time_distributed_5/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0+^time_distributed_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_320752q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_5/StatefulPartitionedCall:output:0,time_distributed_6_time_distributed_6_kernel*time_distributed_6_time_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320724q
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????*   ?
time_distributed_6/ReshapeReshape3time_distributed_5/StatefulPartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
cropping1d/PartitionedCallPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_cropping1d_layer_call_and_return_conditional_losses_320673v
IdentityIdentity#cropping1d/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall+^time_distributed_5/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_5/StatefulPartitionedCall*time_distributed_5/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
3__inference_time_distributed_1_layer_call_fn_324495

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_320860t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_320098

inputs3
!dense_3_time_distributed_6_kernel:*-
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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
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
C__inference_dense_3_layer_call_and_return_conditional_losses_320056\
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
$:??????????????????*: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_320696

lstm_input(
lstm_lstm_cell_kernel:	?
3
lstm_lstm_cell_recurrent_kernel:
??
"
lstm_lstm_cell_bias:	?
-
lstm_1_lstm_cell_1_kernel:
??
7
#lstm_1_lstm_cell_1_recurrent_kernel:
??
&
lstm_1_lstm_cell_1_bias:	?
-
lstm_2_lstm_cell_2_kernel:
??
7
#lstm_2_lstm_cell_2_recurrent_kernel:
??
&
lstm_2_lstm_cell_2_bias:	?
+
time_distributed_kernel:
??$
time_distributed_bias:	?,
time_distributed_2_kernel:	?T%
time_distributed_2_bias:T+
time_distributed_4_kernel:T*%
time_distributed_4_bias:*+
time_distributed_6_kernel:*%
time_distributed_6_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_lstm_cell_kernellstm_lstm_cell_recurrent_kernellstm_lstm_cell_biaslstm_1_lstm_cell_1_kernel#lstm_1_lstm_cell_1_recurrent_kernellstm_1_lstm_cell_1_biaslstm_2_lstm_cell_2_kernel#lstm_2_lstm_cell_2_recurrent_kernellstm_2_lstm_cell_2_biastime_distributed_kerneltime_distributed_biastime_distributed_2_kerneltime_distributed_2_biastime_distributed_4_kerneltime_distributed_4_biastime_distributed_6_kerneltime_distributed_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_320676s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:?????????
$
_user_specified_name
lstm_input
? 
?
while_body_319206
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
'lstm_cell_1_lstm_1_lstm_cell_1_kernel_0:
??
E
1lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
4
%lstm_cell_1_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
%lstm_cell_1_lstm_1_lstm_cell_1_kernel:
??
C
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
2
#lstm_cell_1_lstm_1_lstm_cell_1_bias:	?
??#lstm_cell_1/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'lstm_cell_1_lstm_1_lstm_cell_1_kernel_01lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0%lstm_cell_1_lstm_1_lstm_cell_1_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319153?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_1/StatefulPartitionedCall:output:0*
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
: ~

Identity_4Identity,lstm_cell_1/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_5Identity,lstm_cell_1/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"L
#lstm_cell_1_lstm_1_lstm_cell_1_bias%lstm_cell_1_lstm_1_lstm_cell_1_bias_0"P
%lstm_cell_1_lstm_1_lstm_cell_1_kernel'lstm_cell_1_lstm_1_lstm_cell_1_kernel_0"d
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel1lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?3
?	
#sequential_lstm_2_while_body_318479(
$sequential_lstm_2_while_loop_counter.
*sequential_lstm_2_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#sequential_lstm_2_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_sequential_lstm_2_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!sequential_lstm_2_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_2_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_sequential_lstm_2_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
value	B :g
add_1AddV2$sequential_lstm_2_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: j

Identity_1Identity*sequential_lstm_2_while_maximum_iterations^NoOp*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"H
!sequential_lstm_2_strided_slice_1#sequential_lstm_2_strided_slice_1_0"?
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_2_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_sequential_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_319031
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_319031___redundant_placeholder0.
*while_cond_319031___redundant_placeholder1.
*while_cond_319031___redundant_placeholder2.
*while_cond_319031___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324811

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
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
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*7
_input_shapes&
$:??????????????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????T
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324875

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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*b
dropout_2/IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????*\
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
value	B :*?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshapedropout_2/Identity:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????*"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????*:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?1
?
while_body_324295
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0:
??
]
Ilstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0:
??
K
<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorO
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel:
??
[
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel:
??
I
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias:	?
??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0* 
_output_shapes
:
??
*
dtype0?
lstm_cell_2/MatMul_1MatMulplaceholder_2+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0*
_output_shapes	
:?
*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????s
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell_2/mul_2:z:0*
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
: g

Identity_4Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_5Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"z
:lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias<lstm_cell_2_biasadd_readvariableop_lstm_2_lstm_cell_2_bias_0"?
Glstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernelIlstm_cell_2_matmul_1_readvariableop_lstm_2_lstm_cell_2_recurrent_kernel_0"|
;lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel=lstm_cell_2_matmul_readvariableop_lstm_2_lstm_cell_2_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_time_distributed_1_layer_call_fn_324480

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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_319697n
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
?
?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324993

inputsI
7dense_3_matmul_readvariableop_time_distributed_6_kernel:*D
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
valueB"????*   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????*?
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_time_distributed_6_kernel*
_output_shapes

:**
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
$:??????????????????*: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????*
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324825

inputsI
7dense_2_matmul_readvariableop_time_distributed_4_kernel:T*D
6dense_2_biasadd_readvariableop_time_distributed_4_bias:*
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????T   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0?
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   *   ~
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????*e
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????*?
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:?????????T: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_325372

inputsB
/matmul_readvariableop_time_distributed_2_kernel:	?T<
.biasadd_readvariableop_time_distributed_2_bias:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_2_kernel*
_output_shapes
:	?T*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_2_bias*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Tw
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
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_325431

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????*[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????*"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????*:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?9
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_319420

inputs9
%lstm_cell_2_lstm_2_lstm_cell_2_kernel:
??
C
/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel:
??
2
#lstm_cell_2_lstm_2_lstm_cell_2_bias:	?

identity??#lstm_cell_2/StatefulPartitionedCall?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0%lstm_cell_2_lstm_2_lstm_cell_2_kernel/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel#lstm_cell_2_lstm_2_lstm_cell_2_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_319343n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_cell_2_lstm_2_lstm_cell_2_kernel/lstm_cell_2_lstm_2_lstm_cell_2_recurrent_kernel#lstm_cell_2_lstm_2_lstm_cell_2_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_319354*
condR
while_cond_319353*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325223

inputs
states_0
states_1C
/matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
O
;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
=
.biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
MatMul_1/ReadVariableOpReadVariableOp;matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????\
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????Y
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*U
_input_shapesD
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?I
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323789

inputsO
;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel:
??
[
Glstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:
??
I
:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias:	?

identity??"lstm_cell_1/BiasAdd/ReadVariableOp?!lstm_cell_1/MatMul/ReadVariableOp?#lstm_cell_1/MatMul_1/ReadVariableOp?while;
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
B :?s
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
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
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
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
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
valueB"????Q  ?
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel* 
_output_shapes
:
??
*
dtype0?
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
?
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:??????????
?
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
_output_shapes	
:?
*
dtype0?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????v
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????y
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0;lstm_cell_1_matmul_readvariableop_lstm_1_lstm_cell_1_kernelGlstm_cell_1_matmul_1_readvariableop_lstm_1_lstm_cell_1_recurrent_kernel:lstm_cell_1_biasadd_readvariableop_lstm_1_lstm_cell_1_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323707*
condR
while_cond_323706*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*1
_input_shapes 
:??????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_319912

inputsA
/matmul_readvariableop_time_distributed_4_kernel:T*<
.biasadd_readvariableop_time_distributed_4_bias:*
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_time_distributed_4_kernel*
_output_shapes

:T**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_time_distributed_4_bias*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????*w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
? 
?
while_body_319032
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
'lstm_cell_1_lstm_1_lstm_cell_1_kernel_0:
??
E
1lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0:
??
4
%lstm_cell_1_lstm_1_lstm_cell_1_bias_0:	?

identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
%lstm_cell_1_lstm_1_lstm_cell_1_kernel:
??
C
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel:
??
2
#lstm_cell_1_lstm_1_lstm_cell_1_bias:	?
??#lstm_cell_1/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????Q  ?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3'lstm_cell_1_lstm_1_lstm_cell_1_kernel_01lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0%lstm_cell_1_lstm_1_lstm_cell_1_bias_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_319021?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder,lstm_cell_1/StatefulPartitionedCall:output:0*
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
: ~

Identity_4Identity,lstm_cell_1/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_5Identity,lstm_cell_1/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"L
#lstm_cell_1_lstm_1_lstm_cell_1_bias%lstm_cell_1_lstm_1_lstm_cell_1_bias_0"P
%lstm_cell_1_lstm_1_lstm_cell_1_kernel'lstm_cell_1_lstm_1_lstm_cell_1_kernel_0"d
/lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel1lstm_cell_1_lstm_1_lstm_cell_1_recurrent_kernel_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall: 
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
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: "?L
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
serving_default_lstm_input:0?????????B

cropping1d4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
	$layer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	)layer
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	.layer
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	3layer
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	8layer
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	=layer
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	Blayer
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Kiter
	Ldecay
Mlearning_rateNaccumulator?Oaccumulator?Paccumulator?Qaccumulator?Raccumulator?Saccumulator?Taccumulator?Uaccumulator?Vaccumulator?Waccumulator?Xaccumulator?Yaccumulator?Zaccumulator?[accumulator?\accumulator?]accumulator?^accumulator?"
	optimizer
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16"
trackable_list_wrapper
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
d
state_size

Nkernel
Orecurrent_kernel
Pbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

istates
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
o
state_size

Qkernel
Rrecurrent_kernel
Sbias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

tstates
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
z
state_size

Tkernel
Urecurrent_kernel
Vbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

states
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Wkernel
Xbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
*	variables
+trainable_variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Ykernel
Zbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
4	variables
5trainable_variables
6regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

[kernel
\bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
>	variables
?trainable_variables
@regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

]kernel
^bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adagrad/iter
 : (2training/Adagrad/decay
(:& (2training/Adagrad/learning_rate
(:&	?
2lstm/lstm_cell/kernel
3:1
??
2lstm/lstm_cell/recurrent_kernel
": ?
2lstm/lstm_cell/bias
-:+
??
2lstm_1/lstm_cell_1/kernel
7:5
??
2#lstm_1/lstm_cell_1/recurrent_kernel
&:$?
2lstm_1/lstm_cell_1/bias
-:+
??
2lstm_2/lstm_cell_2/kernel
7:5
??
2#lstm_2/lstm_cell_2/recurrent_kernel
&:$?
2lstm_2/lstm_cell_2/bias
+:)
??2time_distributed/kernel
$:"?2time_distributed/bias
,:*	?T2time_distributed_2/kernel
%:#T2time_distributed_2/bias
+:)T*2time_distributed_4/kernel
%:#*2time_distributed_4/bias
+:)*2time_distributed_6/kernel
%:#2time_distributed_6/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
N0
O1
P2"
trackable_list_wrapper
5
N0
O1
P2"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
{	variables
|trainable_variables
}regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
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
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
$0"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
.0"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
80"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
B0"
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
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
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
?	variables"
_generic_user_object
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
C:A	?
22training/Adagrad/lstm/lstm_cell/kernel/accumulator
N:L
??
2<training/Adagrad/lstm/lstm_cell/recurrent_kernel/accumulator
=:;?
20training/Adagrad/lstm/lstm_cell/bias/accumulator
H:F
??
26training/Adagrad/lstm_1/lstm_cell_1/kernel/accumulator
R:P
??
2@training/Adagrad/lstm_1/lstm_cell_1/recurrent_kernel/accumulator
A:??
24training/Adagrad/lstm_1/lstm_cell_1/bias/accumulator
H:F
??
26training/Adagrad/lstm_2/lstm_cell_2/kernel/accumulator
R:P
??
2@training/Adagrad/lstm_2/lstm_cell_2/recurrent_kernel/accumulator
A:??
24training/Adagrad/lstm_2/lstm_cell_2/bias/accumulator
F:D
??24training/Adagrad/time_distributed/kernel/accumulator
?:=?22training/Adagrad/time_distributed/bias/accumulator
G:E	?T26training/Adagrad/time_distributed_2/kernel/accumulator
@:>T24training/Adagrad/time_distributed_2/bias/accumulator
F:DT*26training/Adagrad/time_distributed_4/kernel/accumulator
@:>*24training/Adagrad/time_distributed_4/bias/accumulator
F:D*26training/Adagrad/time_distributed_6/kernel/accumulator
@:>24training/Adagrad/time_distributed_6/bias/accumulator
?2?
+__inference_sequential_layer_call_fn_320696
+__inference_sequential_layer_call_fn_321606
+__inference_sequential_layer_call_fn_321628
+__inference_sequential_layer_call_fn_321468?
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
F__inference_sequential_layer_call_and_return_conditional_losses_322110
F__inference_sequential_layer_call_and_return_conditional_losses_322613
F__inference_sequential_layer_call_and_return_conditional_losses_321514
F__inference_sequential_layer_call_and_return_conditional_losses_321560?
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
!__inference__wrapped_model_318634
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
%__inference_lstm_layer_call_fn_322621
%__inference_lstm_layer_call_fn_322629
%__inference_lstm_layer_call_fn_322637
%__inference_lstm_layer_call_fn_322645?
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
@__inference_lstm_layer_call_and_return_conditional_losses_322784
@__inference_lstm_layer_call_and_return_conditional_losses_322923
@__inference_lstm_layer_call_and_return_conditional_losses_323062
@__inference_lstm_layer_call_and_return_conditional_losses_323201?
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
?2?
'__inference_lstm_1_layer_call_fn_323209
'__inference_lstm_1_layer_call_fn_323217
'__inference_lstm_1_layer_call_fn_323225
'__inference_lstm_1_layer_call_fn_323233?
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
B__inference_lstm_1_layer_call_and_return_conditional_losses_323372
B__inference_lstm_1_layer_call_and_return_conditional_losses_323511
B__inference_lstm_1_layer_call_and_return_conditional_losses_323650
B__inference_lstm_1_layer_call_and_return_conditional_losses_323789?
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
?2?
'__inference_lstm_2_layer_call_fn_323797
'__inference_lstm_2_layer_call_fn_323805
'__inference_lstm_2_layer_call_fn_323813
'__inference_lstm_2_layer_call_fn_323821?
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
B__inference_lstm_2_layer_call_and_return_conditional_losses_323960
B__inference_lstm_2_layer_call_and_return_conditional_losses_324099
B__inference_lstm_2_layer_call_and_return_conditional_losses_324238
B__inference_lstm_2_layer_call_and_return_conditional_losses_324377?
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
1__inference_time_distributed_layer_call_fn_324384
1__inference_time_distributed_layer_call_fn_324391
1__inference_time_distributed_layer_call_fn_324398
1__inference_time_distributed_layer_call_fn_324405?
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_324426
L__inference_time_distributed_layer_call_and_return_conditional_losses_324447
L__inference_time_distributed_layer_call_and_return_conditional_losses_324461
L__inference_time_distributed_layer_call_and_return_conditional_losses_324475?
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
3__inference_time_distributed_1_layer_call_fn_324480
3__inference_time_distributed_1_layer_call_fn_324485
3__inference_time_distributed_1_layer_call_fn_324490
3__inference_time_distributed_1_layer_call_fn_324495?
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324511
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324534
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324543
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324559?
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
3__inference_time_distributed_2_layer_call_fn_324566
3__inference_time_distributed_2_layer_call_fn_324573
3__inference_time_distributed_2_layer_call_fn_324580
3__inference_time_distributed_2_layer_call_fn_324587?
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324608
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324629
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324643
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324657?
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
3__inference_time_distributed_3_layer_call_fn_324662
3__inference_time_distributed_3_layer_call_fn_324667
3__inference_time_distributed_3_layer_call_fn_324672
3__inference_time_distributed_3_layer_call_fn_324677?
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324693
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324716
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324725
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324741?
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
3__inference_time_distributed_4_layer_call_fn_324748
3__inference_time_distributed_4_layer_call_fn_324755
3__inference_time_distributed_4_layer_call_fn_324762
3__inference_time_distributed_4_layer_call_fn_324769?
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
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324790
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324811
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324825
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324839?
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
3__inference_time_distributed_5_layer_call_fn_324844
3__inference_time_distributed_5_layer_call_fn_324849
3__inference_time_distributed_5_layer_call_fn_324854
3__inference_time_distributed_5_layer_call_fn_324859?
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
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324875
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324898
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324907
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324923?
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
3__inference_time_distributed_6_layer_call_fn_324930
3__inference_time_distributed_6_layer_call_fn_324937
3__inference_time_distributed_6_layer_call_fn_324944
3__inference_time_distributed_6_layer_call_fn_324951?
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
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324972
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324993
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325007
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325021?
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
+__inference_cropping1d_layer_call_fn_325026
+__inference_cropping1d_layer_call_fn_325031?
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
F__inference_cropping1d_layer_call_and_return_conditional_losses_325039
F__inference_cropping1d_layer_call_and_return_conditional_losses_325047?
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
$__inference_signature_wrapper_321584
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
*__inference_lstm_cell_layer_call_fn_325061
*__inference_lstm_cell_layer_call_fn_325075?
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
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325105
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325135?
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
,__inference_lstm_cell_1_layer_call_fn_325149
,__inference_lstm_cell_1_layer_call_fn_325163?
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
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325193
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325223?
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
,__inference_lstm_cell_2_layer_call_fn_325237
,__inference_lstm_cell_2_layer_call_fn_325251?
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
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325281
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325311?
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
&__inference_dense_layer_call_fn_325318?
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
A__inference_dense_layer_call_and_return_conditional_losses_325328?
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
(__inference_dropout_layer_call_fn_325333
(__inference_dropout_layer_call_fn_325338?
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
C__inference_dropout_layer_call_and_return_conditional_losses_325343
C__inference_dropout_layer_call_and_return_conditional_losses_325355?
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
(__inference_dense_1_layer_call_fn_325362?
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
C__inference_dense_1_layer_call_and_return_conditional_losses_325372?
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
*__inference_dropout_1_layer_call_fn_325377
*__inference_dropout_1_layer_call_fn_325382?
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
E__inference_dropout_1_layer_call_and_return_conditional_losses_325387
E__inference_dropout_1_layer_call_and_return_conditional_losses_325399?
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
(__inference_dense_2_layer_call_fn_325406?
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
C__inference_dense_2_layer_call_and_return_conditional_losses_325416?
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
*__inference_dropout_2_layer_call_fn_325421
*__inference_dropout_2_layer_call_fn_325426?
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
E__inference_dropout_2_layer_call_and_return_conditional_losses_325431
E__inference_dropout_2_layer_call_and_return_conditional_losses_325443?
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
(__inference_dense_3_layer_call_fn_325450?
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
C__inference_dense_3_layer_call_and_return_conditional_losses_325460?
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
!__inference__wrapped_model_318634?NOPQRSTUVWXYZ[\]^7?4
-?*
(?%

lstm_input?????????
? ";?8
6

cropping1d(?%

cropping1d??????????
F__inference_cropping1d_layer_call_and_return_conditional_losses_325039?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
F__inference_cropping1d_layer_call_and_return_conditional_losses_325047`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_cropping1d_layer_call_fn_325026wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
+__inference_cropping1d_layer_call_fn_325031S3?0
)?&
$?!
inputs?????????
? "???????????
C__inference_dense_1_layer_call_and_return_conditional_losses_325372]YZ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????T
? |
(__inference_dense_1_layer_call_fn_325362PYZ0?-
&?#
!?
inputs??????????
? "??????????T?
C__inference_dense_2_layer_call_and_return_conditional_losses_325416\[\/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????*
? {
(__inference_dense_2_layer_call_fn_325406O[\/?,
%?"
 ?
inputs?????????T
? "??????????*?
C__inference_dense_3_layer_call_and_return_conditional_losses_325460\]^/?,
%?"
 ?
inputs?????????*
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_325450O]^/?,
%?"
 ?
inputs?????????*
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_325328^WX0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_layer_call_fn_325318QWX0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_325387\3?0
)?&
 ?
inputs?????????T
p 
? "%?"
?
0?????????T
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_325399\3?0
)?&
 ?
inputs?????????T
p
? "%?"
?
0?????????T
? }
*__inference_dropout_1_layer_call_fn_325377O3?0
)?&
 ?
inputs?????????T
p 
? "??????????T}
*__inference_dropout_1_layer_call_fn_325382O3?0
)?&
 ?
inputs?????????T
p
? "??????????T?
E__inference_dropout_2_layer_call_and_return_conditional_losses_325431\3?0
)?&
 ?
inputs?????????*
p 
? "%?"
?
0?????????*
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_325443\3?0
)?&
 ?
inputs?????????*
p
? "%?"
?
0?????????*
? }
*__inference_dropout_2_layer_call_fn_325421O3?0
)?&
 ?
inputs?????????*
p 
? "??????????*}
*__inference_dropout_2_layer_call_fn_325426O3?0
)?&
 ?
inputs?????????*
p
? "??????????*?
C__inference_dropout_layer_call_and_return_conditional_losses_325343^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_325355^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_325333Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_325338Q4?1
*?'
!?
inputs??????????
p
? "????????????
B__inference_lstm_1_layer_call_and_return_conditional_losses_323372?QRSP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323511?QRSP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323650sQRS@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_323789sQRS@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
'__inference_lstm_1_layer_call_fn_323209QRSP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
'__inference_lstm_1_layer_call_fn_323217QRSP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
'__inference_lstm_1_layer_call_fn_323225fQRS@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
'__inference_lstm_1_layer_call_fn_323233fQRS@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
B__inference_lstm_2_layer_call_and_return_conditional_losses_323960?TUVP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324099?TUVP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324238sTUV@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_324377sTUV@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
'__inference_lstm_2_layer_call_fn_323797TUVP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
'__inference_lstm_2_layer_call_fn_323805TUVP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
'__inference_lstm_2_layer_call_fn_323813fTUV@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
'__inference_lstm_2_layer_call_fn_323821fTUV@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325193?QRS???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_325223?QRS???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
,__inference_lstm_cell_1_layer_call_fn_325149?QRS???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
,__inference_lstm_cell_1_layer_call_fn_325163?QRS???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325281?TUV???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_325311?TUV???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
,__inference_lstm_cell_2_layer_call_fn_325237?TUV???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
,__inference_lstm_cell_2_layer_call_fn_325251?TUV???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325105?NOP??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_325135?NOP??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
*__inference_lstm_cell_layer_call_fn_325061?NOP??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
*__inference_lstm_cell_layer_call_fn_325075?NOP??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
@__inference_lstm_layer_call_and_return_conditional_losses_322784?NOPO?L
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
0???????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_322923?NOPO?L
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
0???????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_323062rNOP??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_323201rNOP??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
%__inference_lstm_layer_call_fn_322621~NOPO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
%__inference_lstm_layer_call_fn_322629~NOPO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
%__inference_lstm_layer_call_fn_322637eNOP??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
%__inference_lstm_layer_call_fn_322645eNOP??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
F__inference_sequential_layer_call_and_return_conditional_losses_321514NOPQRSTUVWXYZ[\]^??<
5?2
(?%

lstm_input?????????
p 

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_321560NOPQRSTUVWXYZ[\]^??<
5?2
(?%

lstm_input?????????
p

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_322110{NOPQRSTUVWXYZ[\]^;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_322613{NOPQRSTUVWXYZ[\]^;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
+__inference_sequential_layer_call_fn_320696rNOPQRSTUVWXYZ[\]^??<
5?2
(?%

lstm_input?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_321468rNOPQRSTUVWXYZ[\]^??<
5?2
(?%

lstm_input?????????
p

 
? "???????????
+__inference_sequential_layer_call_fn_321606nNOPQRSTUVWXYZ[\]^;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_321628nNOPQRSTUVWXYZ[\]^;?8
1?.
$?!
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_321584?NOPQRSTUVWXYZ[\]^E?B
? 
;?8
6

lstm_input(?%

lstm_input?????????";?8
6

cropping1d(?%

cropping1d??????????
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324511|E?B
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324534|E?B
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324543j<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_324559j<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
3__inference_time_distributed_1_layer_call_fn_324480oE?B
;?8
.?+
inputs???????????????????
p 

 
? "&?#????????????????????
3__inference_time_distributed_1_layer_call_fn_324485oE?B
;?8
.?+
inputs???????????????????
p

 
? "&?#????????????????????
3__inference_time_distributed_1_layer_call_fn_324490]<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
3__inference_time_distributed_1_layer_call_fn_324495]<?9
2?/
%?"
inputs??????????
p

 
? "????????????
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324608YZE?B
;?8
.?+
inputs???????????????????
p 

 
? "2?/
(?%
0??????????????????T
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324629YZE?B
;?8
.?+
inputs???????????????????
p

 
? "2?/
(?%
0??????????????????T
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324643mYZ<?9
2?/
%?"
inputs??????????
p 

 
? ")?&
?
0?????????T
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_324657mYZ<?9
2?/
%?"
inputs??????????
p

 
? ")?&
?
0?????????T
? ?
3__inference_time_distributed_2_layer_call_fn_324566rYZE?B
;?8
.?+
inputs???????????????????
p 

 
? "%?"??????????????????T?
3__inference_time_distributed_2_layer_call_fn_324573rYZE?B
;?8
.?+
inputs???????????????????
p

 
? "%?"??????????????????T?
3__inference_time_distributed_2_layer_call_fn_324580`YZ<?9
2?/
%?"
inputs??????????
p 

 
? "??????????T?
3__inference_time_distributed_2_layer_call_fn_324587`YZ<?9
2?/
%?"
inputs??????????
p

 
? "??????????T?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324693zD?A
:?7
-?*
inputs??????????????????T
p 

 
? "2?/
(?%
0??????????????????T
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324716zD?A
:?7
-?*
inputs??????????????????T
p

 
? "2?/
(?%
0??????????????????T
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324725h;?8
1?.
$?!
inputs?????????T
p 

 
? ")?&
?
0?????????T
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_324741h;?8
1?.
$?!
inputs?????????T
p

 
? ")?&
?
0?????????T
? ?
3__inference_time_distributed_3_layer_call_fn_324662mD?A
:?7
-?*
inputs??????????????????T
p 

 
? "%?"??????????????????T?
3__inference_time_distributed_3_layer_call_fn_324667mD?A
:?7
-?*
inputs??????????????????T
p

 
? "%?"??????????????????T?
3__inference_time_distributed_3_layer_call_fn_324672[;?8
1?.
$?!
inputs?????????T
p 

 
? "??????????T?
3__inference_time_distributed_3_layer_call_fn_324677[;?8
1?.
$?!
inputs?????????T
p

 
? "??????????T?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324790~[\D?A
:?7
-?*
inputs??????????????????T
p 

 
? "2?/
(?%
0??????????????????*
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324811~[\D?A
:?7
-?*
inputs??????????????????T
p

 
? "2?/
(?%
0??????????????????*
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324825l[\;?8
1?.
$?!
inputs?????????T
p 

 
? ")?&
?
0?????????*
? ?
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_324839l[\;?8
1?.
$?!
inputs?????????T
p

 
? ")?&
?
0?????????*
? ?
3__inference_time_distributed_4_layer_call_fn_324748q[\D?A
:?7
-?*
inputs??????????????????T
p 

 
? "%?"??????????????????*?
3__inference_time_distributed_4_layer_call_fn_324755q[\D?A
:?7
-?*
inputs??????????????????T
p

 
? "%?"??????????????????*?
3__inference_time_distributed_4_layer_call_fn_324762_[\;?8
1?.
$?!
inputs?????????T
p 

 
? "??????????*?
3__inference_time_distributed_4_layer_call_fn_324769_[\;?8
1?.
$?!
inputs?????????T
p

 
? "??????????*?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324875zD?A
:?7
-?*
inputs??????????????????*
p 

 
? "2?/
(?%
0??????????????????*
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324898zD?A
:?7
-?*
inputs??????????????????*
p

 
? "2?/
(?%
0??????????????????*
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324907h;?8
1?.
$?!
inputs?????????*
p 

 
? ")?&
?
0?????????*
? ?
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_324923h;?8
1?.
$?!
inputs?????????*
p

 
? ")?&
?
0?????????*
? ?
3__inference_time_distributed_5_layer_call_fn_324844mD?A
:?7
-?*
inputs??????????????????*
p 

 
? "%?"??????????????????*?
3__inference_time_distributed_5_layer_call_fn_324849mD?A
:?7
-?*
inputs??????????????????*
p

 
? "%?"??????????????????*?
3__inference_time_distributed_5_layer_call_fn_324854[;?8
1?.
$?!
inputs?????????*
p 

 
? "??????????*?
3__inference_time_distributed_5_layer_call_fn_324859[;?8
1?.
$?!
inputs?????????*
p

 
? "??????????*?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324972~]^D?A
:?7
-?*
inputs??????????????????*
p 

 
? "2?/
(?%
0??????????????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_324993~]^D?A
:?7
-?*
inputs??????????????????*
p

 
? "2?/
(?%
0??????????????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325007l]^;?8
1?.
$?!
inputs?????????*
p 

 
? ")?&
?
0?????????
? ?
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_325021l]^;?8
1?.
$?!
inputs?????????*
p

 
? ")?&
?
0?????????
? ?
3__inference_time_distributed_6_layer_call_fn_324930q]^D?A
:?7
-?*
inputs??????????????????*
p 

 
? "%?"???????????????????
3__inference_time_distributed_6_layer_call_fn_324937q]^D?A
:?7
-?*
inputs??????????????????*
p

 
? "%?"???????????????????
3__inference_time_distributed_6_layer_call_fn_324944_]^;?8
1?.
$?!
inputs?????????*
p 

 
? "???????????
3__inference_time_distributed_6_layer_call_fn_324951_]^;?8
1?.
$?!
inputs?????????*
p

 
? "???????????
L__inference_time_distributed_layer_call_and_return_conditional_losses_324426?WXE?B
;?8
.?+
inputs???????????????????
p 

 
? "3?0
)?&
0???????????????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324447?WXE?B
;?8
.?+
inputs???????????????????
p

 
? "3?0
)?&
0???????????????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324461nWX<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_324475nWX<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
1__inference_time_distributed_layer_call_fn_324384sWXE?B
;?8
.?+
inputs???????????????????
p 

 
? "&?#????????????????????
1__inference_time_distributed_layer_call_fn_324391sWXE?B
;?8
.?+
inputs???????????????????
p

 
? "&?#????????????????????
1__inference_time_distributed_layer_call_fn_324398aWX<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
1__inference_time_distributed_layer_call_fn_324405aWX<?9
2?/
%?"
inputs??????????
p

 
? "???????????