╣Њ
жК
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
џ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

┐
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

П
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Г
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
.
Neg
x"T
y"T"
Ttype:

2	
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
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
@
Softplus
features"T
activations"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	љ
Ш
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ
9
VarIsInitializedOp
resource
is_initialized
ѕ*1.15.22v1.15.0-92-g5d80e1e8Ађ
~
PlaceholderPlaceholder*
dtype0*/
_output_shapes
:         *$
shape:         
╣
:unbottleneck/dense/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@unbottleneck/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ф
8unbottleneck/dense/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@unbottleneck/dense/kernel*
valueB
 *JQ┌й*
dtype0*
_output_shapes
: 
Ф
8unbottleneck/dense/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@unbottleneck/dense/kernel*
valueB
 *JQ┌=*
dtype0*
_output_shapes
: 
З
Bunbottleneck/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform:unbottleneck/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*
T0*,
_class"
 loc:@unbottleneck/dense/kernel
ѓ
8unbottleneck/dense/kernel/Initializer/random_uniform/subSub8unbottleneck/dense/kernel/Initializer/random_uniform/max8unbottleneck/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*,
_class"
 loc:@unbottleneck/dense/kernel
Ћ
8unbottleneck/dense/kernel/Initializer/random_uniform/mulMulBunbottleneck/dense/kernel/Initializer/random_uniform/RandomUniform8unbottleneck/dense/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@unbottleneck/dense/kernel*
_output_shapes
:	ђ
Є
4unbottleneck/dense/kernel/Initializer/random_uniformAdd8unbottleneck/dense/kernel/Initializer/random_uniform/mul8unbottleneck/dense/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@unbottleneck/dense/kernel*
_output_shapes
:	ђ
й
unbottleneck/dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	ђ**
shared_nameunbottleneck/dense/kernel*,
_class"
 loc:@unbottleneck/dense/kernel
Ѓ
:unbottleneck/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpunbottleneck/dense/kernel*
_output_shapes
: 
њ
 unbottleneck/dense/kernel/AssignAssignVariableOpunbottleneck/dense/kernel4unbottleneck/dense/kernel/Initializer/random_uniform*
dtype0
ѕ
-unbottleneck/dense/kernel/Read/ReadVariableOpReadVariableOpunbottleneck/dense/kernel*
dtype0*
_output_shapes
:	ђ
ц
)unbottleneck/dense/bias/Initializer/zerosConst**
_class 
loc:@unbottleneck/dense/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
│
unbottleneck/dense/biasVarHandleOp*
shape:ђ*(
shared_nameunbottleneck/dense/bias**
_class 
loc:@unbottleneck/dense/bias*
dtype0*
_output_shapes
: 

8unbottleneck/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpunbottleneck/dense/bias*
_output_shapes
: 
Ѓ
unbottleneck/dense/bias/AssignAssignVariableOpunbottleneck/dense/bias)unbottleneck/dense/bias/Initializer/zeros*
dtype0
ђ
+unbottleneck/dense/bias/Read/ReadVariableOpReadVariableOpunbottleneck/dense/bias*
dtype0*
_output_shapes	
:ђ
є
+unbottleneck/dense/Tensordot/ReadVariableOpReadVariableOpunbottleneck/dense/kernel*
dtype0*
_output_shapes
:	ђ
k
!unbottleneck/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
v
!unbottleneck/dense/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          
]
"unbottleneck/dense/Tensordot/ShapeShapePlaceholder*
T0*
_output_shapes
:
l
*unbottleneck/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
В
%unbottleneck/dense/Tensordot/GatherV2GatherV2"unbottleneck/dense/Tensordot/Shape!unbottleneck/dense/Tensordot/free*unbottleneck/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
n
,unbottleneck/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
'unbottleneck/dense/Tensordot/GatherV2_1GatherV2"unbottleneck/dense/Tensordot/Shape!unbottleneck/dense/Tensordot/axes,unbottleneck/dense/Tensordot/GatherV2_1/axis*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0
l
"unbottleneck/dense/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ћ
!unbottleneck/dense/Tensordot/ProdProd%unbottleneck/dense/Tensordot/GatherV2"unbottleneck/dense/Tensordot/Const*
_output_shapes
: *
T0
n
$unbottleneck/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Џ
#unbottleneck/dense/Tensordot/Prod_1Prod'unbottleneck/dense/Tensordot/GatherV2_1$unbottleneck/dense/Tensordot/Const_1*
T0*
_output_shapes
: 
j
(unbottleneck/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
═
#unbottleneck/dense/Tensordot/concatConcatV2!unbottleneck/dense/Tensordot/free!unbottleneck/dense/Tensordot/axes(unbottleneck/dense/Tensordot/concat/axis*
T0*
N*
_output_shapes
:
а
"unbottleneck/dense/Tensordot/stackPack!unbottleneck/dense/Tensordot/Prod#unbottleneck/dense/Tensordot/Prod_1*
T0*
N*
_output_shapes
:
Ъ
&unbottleneck/dense/Tensordot/transpose	TransposePlaceholder#unbottleneck/dense/Tensordot/concat*
T0*/
_output_shapes
:         
Х
$unbottleneck/dense/Tensordot/ReshapeReshape&unbottleneck/dense/Tensordot/transpose"unbottleneck/dense/Tensordot/stack*
T0*0
_output_shapes
:                  
~
-unbottleneck/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
╗
(unbottleneck/dense/Tensordot/transpose_1	Transpose+unbottleneck/dense/Tensordot/ReadVariableOp-unbottleneck/dense/Tensordot/transpose_1/perm*
_output_shapes
:	ђ*
T0
}
,unbottleneck/dense/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
│
&unbottleneck/dense/Tensordot/Reshape_1Reshape(unbottleneck/dense/Tensordot/transpose_1,unbottleneck/dense/Tensordot/Reshape_1/shape*
T0*
_output_shapes
:	ђ
«
#unbottleneck/dense/Tensordot/MatMulMatMul$unbottleneck/dense/Tensordot/Reshape&unbottleneck/dense/Tensordot/Reshape_1*
T0*(
_output_shapes
:         ђ
o
$unbottleneck/dense/Tensordot/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
l
*unbottleneck/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
п
%unbottleneck/dense/Tensordot/concat_1ConcatV2%unbottleneck/dense/Tensordot/GatherV2$unbottleneck/dense/Tensordot/Const_2*unbottleneck/dense/Tensordot/concat_1/axis*
T0*
N*
_output_shapes
:
«
unbottleneck/dense/TensordotReshape#unbottleneck/dense/Tensordot/MatMul%unbottleneck/dense/Tensordot/concat_1*
T0*0
_output_shapes
:         ђ
~
)unbottleneck/dense/BiasAdd/ReadVariableOpReadVariableOpunbottleneck/dense/bias*
dtype0*
_output_shapes	
:ђ
Е
unbottleneck/dense/BiasAddBiasAddunbottleneck/dense/Tensordot)unbottleneck/dense/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
╦
?decoder/layer_0/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_0/strided/kernel*%
valueB"            *
dtype0*
_output_shapes
:
х
=decoder/layer_0/strided/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@decoder/layer_0/strided/kernel*
valueB
 *q─ю╝*
dtype0*
_output_shapes
: 
х
=decoder/layer_0/strided/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_0/strided/kernel*
valueB
 *q─ю<
ї
Gdecoder/layer_0/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_0/strided/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@decoder/layer_0/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ќ
=decoder/layer_0/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_0/strided/kernel/Initializer/random_uniform/max=decoder/layer_0/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_0/strided/kernel*
_output_shapes
: 
▓
=decoder/layer_0/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_0/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_0/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_0/strided/kernel*(
_output_shapes
:ђђ
ц
9decoder/layer_0/strided/kernel/Initializer/random_uniformAdd=decoder/layer_0/strided/kernel/Initializer/random_uniform/mul=decoder/layer_0/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_0/strided/kernel*(
_output_shapes
:ђђ
Н
decoder/layer_0/strided/kernelVarHandleOp*1
_class'
%#loc:@decoder/layer_0/strided/kernel*
dtype0*
_output_shapes
: *
shape:ђђ*/
shared_name decoder/layer_0/strided/kernel
Ї
?decoder/layer_0/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_0/strided/kernel*
_output_shapes
: 
А
%decoder/layer_0/strided/kernel/AssignAssignVariableOpdecoder/layer_0/strided/kernel9decoder/layer_0/strided/kernel/Initializer/random_uniform*
dtype0
Џ
2decoder/layer_0/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_0/strided/kernel*
dtype0*(
_output_shapes
:ђђ
«
.decoder/layer_0/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_0/strided/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
┬
decoder/layer_0/strided/biasVarHandleOp*
shape:ђ*-
shared_namedecoder/layer_0/strided/bias*/
_class%
#!loc:@decoder/layer_0/strided/bias*
dtype0*
_output_shapes
: 
Ѕ
=decoder/layer_0/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_0/strided/bias*
_output_shapes
: 
њ
#decoder/layer_0/strided/bias/AssignAssignVariableOpdecoder/layer_0/strided/bias.decoder/layer_0/strided/bias/Initializer/zeros*
dtype0
і
0decoder/layer_0/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_0/strided/bias*
dtype0*
_output_shapes	
:ђ
g
decoder/layer_0/strided/ShapeShapeunbottleneck/dense/BiasAdd*
T0*
_output_shapes
:
u
+decoder/layer_0/strided/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-decoder/layer_0/strided/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-decoder/layer_0/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_0/strided/strided_sliceStridedSlicedecoder/layer_0/strided/Shape+decoder/layer_0/strided/strided_slice/stack-decoder/layer_0/strided/strided_slice/stack_1-decoder/layer_0/strided/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_0/strided/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_0/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_0/strided/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Г
'decoder/layer_0/strided/strided_slice_1StridedSlicedecoder/layer_0/strided/Shape-decoder/layer_0/strided/strided_slice_1/stack/decoder/layer_0/strided/strided_slice_1/stack_1/decoder/layer_0/strided/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
w
-decoder/layer_0/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_0/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_0/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_0/strided/strided_slice_2StridedSlicedecoder/layer_0/strided/Shape-decoder/layer_0/strided/strided_slice_2/stack/decoder/layer_0/strided/strided_slice_2/stack_1/decoder/layer_0/strided/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
_
decoder/layer_0/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_0/strided/mulMul'decoder/layer_0/strided/strided_slice_1decoder/layer_0/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_0/strided/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
decoder/layer_0/strided/mul_1Mul'decoder/layer_0/strided/strided_slice_2decoder/layer_0/strided/mul_1/y*
T0*
_output_shapes
: 
b
decoder/layer_0/strided/stack/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
О
decoder/layer_0/strided/stackPack%decoder/layer_0/strided/strided_slicedecoder/layer_0/strided/muldecoder/layer_0/strided/mul_1decoder/layer_0/strided/stack/3*
T0*
N*
_output_shapes
:
w
-decoder/layer_0/strided/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_0/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_0/strided/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Г
'decoder/layer_0/strided/strided_slice_3StridedSlicedecoder/layer_0/strided/stack-decoder/layer_0/strided/strided_slice_3/stack/decoder/layer_0/strided/strided_slice_3/stack_1/decoder/layer_0/strided/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
а
7decoder/layer_0/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_0/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ќ
(decoder/layer_0/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_0/strided/stack7decoder/layer_0/strided/conv2d_transpose/ReadVariableOpunbottleneck/dense/BiasAdd*
strides
*0
_output_shapes
:         ђ*
paddingSAME*
T0
ѕ
.decoder/layer_0/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_0/strided/bias*
dtype0*
_output_shapes	
:ђ
┐
decoder/layer_0/strided/BiasAddBiasAdd(decoder/layer_0/strided/conv2d_transpose.decoder/layer_0/strided/BiasAdd/ReadVariableOp*0
_output_shapes
:         ђ*
T0
n
decoder/layer_0/strided/Shape_1Shapedecoder/layer_0/strided/BiasAdd*
_output_shapes
:*
T0
w
-decoder/layer_0/strided/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/decoder/layer_0/strided/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_0/strided/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
»
'decoder/layer_0/strided/strided_slice_4StridedSlicedecoder/layer_0/strided/Shape_1-decoder/layer_0/strided/strided_slice_4/stack/decoder/layer_0/strided/strided_slice_4/stack_1/decoder/layer_0/strided/strided_slice_4/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
i
'decoder/layer_0/strided/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
i
'decoder/layer_0/strided/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
r
'decoder/layer_0/strided/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
valueB :
         
i
'decoder/layer_0/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_0/strided/Reshape/shapePack'decoder/layer_0/strided/strided_slice_4'decoder/layer_0/strided/Reshape/shape/1'decoder/layer_0/strided/Reshape/shape/2'decoder/layer_0/strided/Reshape/shape/3'decoder/layer_0/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_0/strided/ReshapeReshapedecoder/layer_0/strided/BiasAdd%decoder/layer_0/strided/Reshape/shape*
T0*<
_output_shapes*
(:&                  
_
decoder/layer_0/strided/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_0/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_0/strided/splitSplit'decoder/layer_0/strided/split/split_dimdecoder/layer_0/strided/Reshape*
	num_split*d
_output_shapesR
P:&                  :&                  *
T0
ѕ
decoder/layer_0/strided/EluEludecoder/layer_0/strided/split*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_0/strided/NegNegdecoder/layer_0/strided/split:1*<
_output_shapes*
(:&                  *
T0
ѕ
decoder/layer_0/strided/Elu_1Eludecoder/layer_0/strided/Neg*<
_output_shapes*
(:&                  *
T0
і
decoder/layer_0/strided/Neg_1Negdecoder/layer_0/strided/Elu_1*
T0*<
_output_shapes*
(:&                  
n
#decoder/layer_0/strided/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
decoder/layer_0/strided/concatConcatV2decoder/layer_0/strided/Eludecoder/layer_0/strided/Neg_1#decoder/layer_0/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
k
)decoder/layer_0/strided/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
k
)decoder/layer_0/strided/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
l
)decoder/layer_0/strided/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
Є
'decoder/layer_0/strided/Reshape_1/shapePack'decoder/layer_0/strided/strided_slice_4)decoder/layer_0/strided/Reshape_1/shape/1)decoder/layer_0/strided/Reshape_1/shape/2)decoder/layer_0/strided/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
░
!decoder/layer_0/strided/Reshape_1Reshapedecoder/layer_0/strided/concat'decoder/layer_0/strided/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*%
valueB"            
¤
Jdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*
valueB
 *Uей*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*
valueB
 *Uе=*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
╩
Jdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_0/residual_0/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_0/residual_0/depthwise_kernel*>
_class4
20loc:@decoder/layer_0/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
Д
Ldecoder/layer_0/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_0/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_0/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_0/residual_0/depthwise_kernelFdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_0/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_0/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*%
valueB"            
¤
Jdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*
valueB
 *  ђй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*
valueB
 *  ђ=*
dtype0*
_output_shapes
: 
│
Tdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
п
Fdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform/min*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel
Ч
+decoder/layer_0/residual_0/pointwise_kernelVarHandleOp*
shape:ђђ*<
shared_name-+decoder/layer_0/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_0/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: 
Д
Ldecoder/layer_0/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_0/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_0/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_0/residual_0/pointwise_kernelFdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_0/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_0/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
└
Adecoder/layer_0/residual_0/bias/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@decoder/layer_0/residual_0/bias*
valueB:ђ*
dtype0*
_output_shapes
:
░
7decoder/layer_0/residual_0/bias/Initializer/zeros/ConstConst*2
_class(
&$loc:@decoder/layer_0/residual_0/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
1decoder/layer_0/residual_0/bias/Initializer/zerosFillAdecoder/layer_0/residual_0/bias/Initializer/zeros/shape_as_tensor7decoder/layer_0/residual_0/bias/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*2
_class(
&$loc:@decoder/layer_0/residual_0/bias
╦
decoder/layer_0/residual_0/biasVarHandleOp*2
_class(
&$loc:@decoder/layer_0/residual_0/bias*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_0/residual_0/bias
Ј
@decoder/layer_0/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_0/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_0/residual_0/bias/AssignAssignVariableOpdecoder/layer_0/residual_0/bias1decoder/layer_0/residual_0/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_0/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_0/residual_0/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_0/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_0/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_0/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_0/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_0/residual_0/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_0/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
љ
5decoder/layer_0/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_0/strided/Reshape_1:decoder/layer_0/residual_0/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ј
+decoder/layer_0/residual_0/separable_conv2dConv2D5decoder/layer_0/residual_0/separable_conv2d/depthwise<decoder/layer_0/residual_0/separable_conv2d/ReadVariableOp_1*
strides
*0
_output_shapes
:         ђ*
paddingVALID*
T0
ј
1decoder/layer_0/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_0/residual_0/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_0/residual_0/BiasAddBiasAdd+decoder/layer_0/residual_0/separable_conv2d1decoder/layer_0/residual_0/BiasAdd/ReadVariableOp*0
_output_shapes
:         ђ*
T0
r
 decoder/layer_0/residual_0/ShapeShape"decoder/layer_0/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_0/residual_0/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
z
0decoder/layer_0/residual_0/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/layer_0/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_0/residual_0/strided_sliceStridedSlice decoder/layer_0/residual_0/Shape.decoder/layer_0/residual_0/strided_slice/stack0decoder/layer_0/residual_0/strided_slice/stack_10decoder/layer_0/residual_0/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
l
*decoder/layer_0/residual_0/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_0/residual_0/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_0/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_0/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_0/residual_0/Reshape/shapePack(decoder/layer_0/residual_0/strided_slice*decoder/layer_0/residual_0/Reshape/shape/1*decoder/layer_0/residual_0/Reshape/shape/2*decoder/layer_0/residual_0/Reshape/shape/3*decoder/layer_0/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_0/residual_0/ReshapeReshape"decoder/layer_0/residual_0/BiasAdd(decoder/layer_0/residual_0/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_0/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_0/residual_0/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_0/residual_0/splitSplit*decoder/layer_0/residual_0/split/split_dim"decoder/layer_0/residual_0/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ј
decoder/layer_0/residual_0/EluElu decoder/layer_0/residual_0/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_0/residual_0/NegNeg"decoder/layer_0/residual_0/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_0/residual_0/Elu_1Eludecoder/layer_0/residual_0/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_0/residual_0/Neg_1Neg decoder/layer_0/residual_0/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_0/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_0/residual_0/concatConcatV2decoder/layer_0/residual_0/Elu decoder/layer_0/residual_0/Neg_1&decoder/layer_0/residual_0/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
n
,decoder/layer_0/residual_0/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_0/residual_0/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
o
,decoder/layer_0/residual_0/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_0/residual_0/Reshape_1/shapePack(decoder/layer_0/residual_0/strided_slice,decoder/layer_0/residual_0/Reshape_1/shape/1,decoder/layer_0/residual_0/Reshape_1/shape/2,decoder/layer_0/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_0/residual_0/Reshape_1Reshape!decoder/layer_0/residual_0/concat*decoder/layer_0/residual_0/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel*
valueB
 *лвл╝*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel*
valueB
 *лвл<*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel
╩
Jdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel
т
Jdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel
О
Fdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_0/residual_1/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_0/residual_1/depthwise_kernel*>
_class4
20loc:@decoder/layer_0/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
Д
Ldecoder/layer_0/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_0/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_0/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_0/residual_1/depthwise_kernelFdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_0/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_0/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*%
valueB"            
¤
Jdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*
valueB
 *  ђй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*
valueB
 *  ђ=*
dtype0*
_output_shapes
: 
│
Tdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/sub*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel
п
Fdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_0/residual_1/pointwise_kernelVarHandleOp*<
shared_name-+decoder/layer_0/residual_1/pointwise_kernel*>
_class4
20loc:@decoder/layer_0/residual_1/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:ђђ
Д
Ldecoder/layer_0/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_0/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_0/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_0/residual_1/pointwise_kernelFdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_0/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_0/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_0/residual_1/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_0/residual_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_0/residual_1/biasVarHandleOp*
shape:ђ*0
shared_name!decoder/layer_0/residual_1/bias*2
_class(
&$loc:@decoder/layer_0/residual_1/bias*
dtype0*
_output_shapes
: 
Ј
@decoder/layer_0/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_0/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_0/residual_1/bias/AssignAssignVariableOpdecoder/layer_0/residual_1/bias1decoder/layer_0/residual_1/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_0/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_0/residual_1/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_0/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_0/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_0/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_0/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_0/residual_1/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_0/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Њ
5decoder/layer_0/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_0/residual_0/Reshape_1:decoder/layer_0/residual_1/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ј
+decoder/layer_0/residual_1/separable_conv2dConv2D5decoder/layer_0/residual_1/separable_conv2d/depthwise<decoder/layer_0/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*0
_output_shapes
:         ђ
ј
1decoder/layer_0/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_0/residual_1/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_0/residual_1/BiasAddBiasAdd+decoder/layer_0/residual_1/separable_conv2d1decoder/layer_0/residual_1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
r
 decoder/layer_0/residual_1/ShapeShape"decoder/layer_0/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_0/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_0/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_0/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_0/residual_1/strided_sliceStridedSlice decoder/layer_0/residual_1/Shape.decoder/layer_0/residual_1/strided_slice/stack0decoder/layer_0/residual_1/strided_slice/stack_10decoder/layer_0/residual_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
l
*decoder/layer_0/residual_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_0/residual_1/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_0/residual_1/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_0/residual_1/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_0/residual_1/Reshape/shapePack(decoder/layer_0/residual_1/strided_slice*decoder/layer_0/residual_1/Reshape/shape/1*decoder/layer_0/residual_1/Reshape/shape/2*decoder/layer_0/residual_1/Reshape/shape/3*decoder/layer_0/residual_1/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_0/residual_1/ReshapeReshape"decoder/layer_0/residual_1/BiasAdd(decoder/layer_0/residual_1/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_0/residual_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_0/residual_1/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_0/residual_1/splitSplit*decoder/layer_0/residual_1/split/split_dim"decoder/layer_0/residual_1/Reshape*
	num_split*d
_output_shapesR
P:&                  :&                  *
T0
ј
decoder/layer_0/residual_1/EluElu decoder/layer_0/residual_1/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_0/residual_1/NegNeg"decoder/layer_0/residual_1/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_0/residual_1/Elu_1Eludecoder/layer_0/residual_1/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_0/residual_1/Neg_1Neg decoder/layer_0/residual_1/Elu_1*<
_output_shapes*
(:&                  *
T0
q
&decoder/layer_0/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_0/residual_1/concatConcatV2decoder/layer_0/residual_1/Elu decoder/layer_0/residual_1/Neg_1&decoder/layer_0/residual_1/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
n
,decoder/layer_0/residual_1/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_0/residual_1/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
o
,decoder/layer_0/residual_1/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value
B :ђ
ћ
*decoder/layer_0/residual_1/Reshape_1/shapePack(decoder/layer_0/residual_1/strided_slice,decoder/layer_0/residual_1/Reshape_1/shape/1,decoder/layer_0/residual_1/Reshape_1/shape/2,decoder/layer_0/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_0/residual_1/Reshape_1Reshape!decoder/layer_0/residual_1/concat*decoder/layer_0/residual_1/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
а
decoder/layer_0/addAddV2!decoder/layer_0/strided/Reshape_1$decoder/layer_0/residual_1/Reshape_1*
T0*0
_output_shapes
:         ђ
X
decoder/layer_0/ShapeShapedecoder/layer_0/add*
T0*
_output_shapes
:
m
#decoder/layer_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
decoder/layer_0/strided_sliceStridedSlicedecoder/layer_0/Shape#decoder/layer_0/strided_slice/stack%decoder/layer_0/strided_slice/stack_1%decoder/layer_0/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
╗
4decoder/layer_0/ln/layer_norm_scale/Initializer/onesConst*6
_class,
*(loc:@decoder/layer_0/ln/layer_norm_scale*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
О
#decoder/layer_0/ln/layer_norm_scaleVarHandleOp*6
_class,
*(loc:@decoder/layer_0/ln/layer_norm_scale*
dtype0*
_output_shapes
: *
shape:ђ*4
shared_name%#decoder/layer_0/ln/layer_norm_scale
Ќ
Ddecoder/layer_0/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_0/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_0/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_0/ln/layer_norm_scale4decoder/layer_0/ln/layer_norm_scale/Initializer/ones*
dtype0
ў
7decoder/layer_0/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_0/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
║
4decoder/layer_0/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_0/ln/layer_norm_bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
н
"decoder/layer_0/ln/layer_norm_biasVarHandleOp*
shape:ђ*3
shared_name$"decoder/layer_0/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_0/ln/layer_norm_bias*
dtype0*
_output_shapes
: 
Ћ
Cdecoder/layer_0/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_0/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_0/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_0/ln/layer_norm_bias4decoder/layer_0/ln/layer_norm_bias/Initializer/zeros*
dtype0
ќ
6decoder/layer_0/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_0/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
]
decoder/layer_0/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
ѓ
!decoder/layer_0/ln/ReadVariableOpReadVariableOp#decoder/layer_0/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
Ѓ
#decoder/layer_0/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_0/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
|
)decoder/layer_0/ln/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
         
ф
decoder/layer_0/ln/MeanMeandecoder/layer_0/add)decoder/layer_0/ln/Mean/reduction_indices*/
_output_shapes
:         *
	keep_dims(*
T0
б
$decoder/layer_0/ln/SquaredDifferenceSquaredDifferencedecoder/layer_0/adddecoder/layer_0/ln/Mean*
T0*0
_output_shapes
:         ђ
~
+decoder/layer_0/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┐
decoder/layer_0/ln/Mean_1Mean$decoder/layer_0/ln/SquaredDifference+decoder/layer_0/ln/Mean_1/reduction_indices*
	keep_dims(*
T0*/
_output_shapes
:         
є
decoder/layer_0/ln/subSubdecoder/layer_0/adddecoder/layer_0/ln/Mean*
T0*0
_output_shapes
:         ђ
ј
decoder/layer_0/ln/addAddV2decoder/layer_0/ln/Mean_1decoder/layer_0/ln/Const*
T0*/
_output_shapes
:         
s
decoder/layer_0/ln/RsqrtRsqrtdecoder/layer_0/ln/add*
T0*/
_output_shapes
:         
і
decoder/layer_0/ln/mulMuldecoder/layer_0/ln/subdecoder/layer_0/ln/Rsqrt*
T0*0
_output_shapes
:         ђ
Ћ
decoder/layer_0/ln/mul_1Muldecoder/layer_0/ln/mul!decoder/layer_0/ln/ReadVariableOp*
T0*0
_output_shapes
:         ђ
Џ
decoder/layer_0/ln/add_1AddV2decoder/layer_0/ln/mul_1#decoder/layer_0/ln/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ
╦
?decoder/layer_1/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_1/strided/kernel*%
valueB"            *
dtype0*
_output_shapes
:
х
=decoder/layer_1/strided/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_1/strided/kernel*
valueB
 *q─ю╝
х
=decoder/layer_1/strided/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@decoder/layer_1/strided/kernel*
valueB
 *q─ю<*
dtype0*
_output_shapes
: 
ї
Gdecoder/layer_1/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_1/strided/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@decoder/layer_1/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ќ
=decoder/layer_1/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_1/strided/kernel/Initializer/random_uniform/max=decoder/layer_1/strided/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*1
_class'
%#loc:@decoder/layer_1/strided/kernel
▓
=decoder/layer_1/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_1/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_1/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_1/strided/kernel*(
_output_shapes
:ђђ
ц
9decoder/layer_1/strided/kernel/Initializer/random_uniformAdd=decoder/layer_1/strided/kernel/Initializer/random_uniform/mul=decoder/layer_1/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_1/strided/kernel*(
_output_shapes
:ђђ
Н
decoder/layer_1/strided/kernelVarHandleOp*1
_class'
%#loc:@decoder/layer_1/strided/kernel*
dtype0*
_output_shapes
: *
shape:ђђ*/
shared_name decoder/layer_1/strided/kernel
Ї
?decoder/layer_1/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_1/strided/kernel*
_output_shapes
: 
А
%decoder/layer_1/strided/kernel/AssignAssignVariableOpdecoder/layer_1/strided/kernel9decoder/layer_1/strided/kernel/Initializer/random_uniform*
dtype0
Џ
2decoder/layer_1/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_1/strided/kernel*
dtype0*(
_output_shapes
:ђђ
«
.decoder/layer_1/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_1/strided/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
┬
decoder/layer_1/strided/biasVarHandleOp*-
shared_namedecoder/layer_1/strided/bias*/
_class%
#!loc:@decoder/layer_1/strided/bias*
dtype0*
_output_shapes
: *
shape:ђ
Ѕ
=decoder/layer_1/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_1/strided/bias*
_output_shapes
: 
њ
#decoder/layer_1/strided/bias/AssignAssignVariableOpdecoder/layer_1/strided/bias.decoder/layer_1/strided/bias/Initializer/zeros*
dtype0
і
0decoder/layer_1/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_1/strided/bias*
dtype0*
_output_shapes	
:ђ
e
decoder/layer_1/strided/ShapeShapedecoder/layer_0/ln/add_1*
_output_shapes
:*
T0
u
+decoder/layer_1/strided/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-decoder/layer_1/strided/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-decoder/layer_1/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_1/strided/strided_sliceStridedSlicedecoder/layer_1/strided/Shape+decoder/layer_1/strided/strided_slice/stack-decoder/layer_1/strided/strided_slice/stack_1-decoder/layer_1/strided/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_1/strided/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_1/strided/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_1/strided/strided_slice_1StridedSlicedecoder/layer_1/strided/Shape-decoder/layer_1/strided/strided_slice_1/stack/decoder/layer_1/strided/strided_slice_1/stack_1/decoder/layer_1/strided/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
w
-decoder/layer_1/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_1/strided/strided_slice_2StridedSlicedecoder/layer_1/strided/Shape-decoder/layer_1/strided/strided_slice_2/stack/decoder/layer_1/strided/strided_slice_2/stack_1/decoder/layer_1/strided/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
_
decoder/layer_1/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_1/strided/mulMul'decoder/layer_1/strided/strided_slice_1decoder/layer_1/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_1/strided/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
decoder/layer_1/strided/mul_1Mul'decoder/layer_1/strided/strided_slice_2decoder/layer_1/strided/mul_1/y*
T0*
_output_shapes
: 
b
decoder/layer_1/strided/stack/3Const*
dtype0*
_output_shapes
: *
value
B :ђ
О
decoder/layer_1/strided/stackPack%decoder/layer_1/strided/strided_slicedecoder/layer_1/strided/muldecoder/layer_1/strided/mul_1decoder/layer_1/strided/stack/3*
T0*
N*
_output_shapes
:
w
-decoder/layer_1/strided/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_1/strided/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_1/strided/strided_slice_3StridedSlicedecoder/layer_1/strided/stack-decoder/layer_1/strided/strided_slice_3/stack/decoder/layer_1/strided/strided_slice_3/stack_1/decoder/layer_1/strided/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
а
7decoder/layer_1/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_1/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ћ
(decoder/layer_1/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_1/strided/stack7decoder/layer_1/strided/conv2d_transpose/ReadVariableOpdecoder/layer_0/ln/add_1*
T0*
strides
*0
_output_shapes
:         ђ*
paddingSAME
ѕ
.decoder/layer_1/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_1/strided/bias*
dtype0*
_output_shapes	
:ђ
┐
decoder/layer_1/strided/BiasAddBiasAdd(decoder/layer_1/strided/conv2d_transpose.decoder/layer_1/strided/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
n
decoder/layer_1/strided/Shape_1Shapedecoder/layer_1/strided/BiasAdd*
T0*
_output_shapes
:
w
-decoder/layer_1/strided/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_1/strided/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
»
'decoder/layer_1/strided/strided_slice_4StridedSlicedecoder/layer_1/strided/Shape_1-decoder/layer_1/strided/strided_slice_4/stack/decoder/layer_1/strided/strided_slice_4/stack_1/decoder/layer_1/strided/strided_slice_4/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
i
'decoder/layer_1/strided/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
i
'decoder/layer_1/strided/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_1/strided/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
i
'decoder/layer_1/strided/Reshape/shape/4Const*
dtype0*
_output_shapes
: *
value	B :
е
%decoder/layer_1/strided/Reshape/shapePack'decoder/layer_1/strided/strided_slice_4'decoder/layer_1/strided/Reshape/shape/1'decoder/layer_1/strided/Reshape/shape/2'decoder/layer_1/strided/Reshape/shape/3'decoder/layer_1/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_1/strided/ReshapeReshapedecoder/layer_1/strided/BiasAdd%decoder/layer_1/strided/Reshape/shape*<
_output_shapes*
(:&                  *
T0
_
decoder/layer_1/strided/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_1/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_1/strided/splitSplit'decoder/layer_1/strided/split/split_dimdecoder/layer_1/strided/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ѕ
decoder/layer_1/strided/EluEludecoder/layer_1/strided/split*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_1/strided/NegNegdecoder/layer_1/strided/split:1*
T0*<
_output_shapes*
(:&                  
ѕ
decoder/layer_1/strided/Elu_1Eludecoder/layer_1/strided/Neg*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_1/strided/Neg_1Negdecoder/layer_1/strided/Elu_1*
T0*<
_output_shapes*
(:&                  
n
#decoder/layer_1/strided/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
█
decoder/layer_1/strided/concatConcatV2decoder/layer_1/strided/Eludecoder/layer_1/strided/Neg_1#decoder/layer_1/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
k
)decoder/layer_1/strided/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
k
)decoder/layer_1/strided/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
l
)decoder/layer_1/strided/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
Є
'decoder/layer_1/strided/Reshape_1/shapePack'decoder/layer_1/strided/strided_slice_4)decoder/layer_1/strided/Reshape_1/shape/1)decoder/layer_1/strided/Reshape_1/shape/2)decoder/layer_1/strided/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
░
!decoder/layer_1/strided/Reshape_1Reshapedecoder/layer_1/strided/concat'decoder/layer_1/strided/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*
valueB
 *Uей*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*
valueB
 *Uе=*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
╩
Jdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_1/residual_0/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_1/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ*<
shared_name-+decoder/layer_1/residual_0/depthwise_kernel
Д
Ldecoder/layer_1/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_1/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_1/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_1/residual_0/depthwise_kernelFdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_1/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_1/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel*
valueB
 *  ђй
¤
Jdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel*
valueB
 *  ђ=
│
Tdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel
╩
Jdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel
Т
Jdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/sub*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel
п
Fdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_1/residual_0/pointwise_kernelVarHandleOp*
shape:ђђ*<
shared_name-+decoder/layer_1/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_1/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: 
Д
Ldecoder/layer_1/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_1/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_1/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_1/residual_0/pointwise_kernelFdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_1/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_1/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
└
Adecoder/layer_1/residual_0/bias/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@decoder/layer_1/residual_0/bias*
valueB:ђ*
dtype0*
_output_shapes
:
░
7decoder/layer_1/residual_0/bias/Initializer/zeros/ConstConst*2
_class(
&$loc:@decoder/layer_1/residual_0/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
1decoder/layer_1/residual_0/bias/Initializer/zerosFillAdecoder/layer_1/residual_0/bias/Initializer/zeros/shape_as_tensor7decoder/layer_1/residual_0/bias/Initializer/zeros/Const*
T0*2
_class(
&$loc:@decoder/layer_1/residual_0/bias*
_output_shapes	
:ђ
╦
decoder/layer_1/residual_0/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_1/residual_0/bias*2
_class(
&$loc:@decoder/layer_1/residual_0/bias
Ј
@decoder/layer_1/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_1/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_1/residual_0/bias/AssignAssignVariableOpdecoder/layer_1/residual_0/bias1decoder/layer_1/residual_0/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_1/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_1/residual_0/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_1/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_1/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_1/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_1/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_1/residual_0/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_1/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
љ
5decoder/layer_1/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_1/strided/Reshape_1:decoder/layer_1/residual_0/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ј
+decoder/layer_1/residual_0/separable_conv2dConv2D5decoder/layer_1/residual_0/separable_conv2d/depthwise<decoder/layer_1/residual_0/separable_conv2d/ReadVariableOp_1*
T0*
strides
*0
_output_shapes
:         ђ*
paddingVALID
ј
1decoder/layer_1/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_1/residual_0/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_1/residual_0/BiasAddBiasAdd+decoder/layer_1/residual_0/separable_conv2d1decoder/layer_1/residual_0/BiasAdd/ReadVariableOp*0
_output_shapes
:         ђ*
T0
r
 decoder/layer_1/residual_0/ShapeShape"decoder/layer_1/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_1/residual_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_1/residual_0/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/layer_1/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_1/residual_0/strided_sliceStridedSlice decoder/layer_1/residual_0/Shape.decoder/layer_1/residual_0/strided_slice/stack0decoder/layer_1/residual_0/strided_slice/stack_10decoder/layer_1/residual_0/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
l
*decoder/layer_1/residual_0/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
l
*decoder/layer_1/residual_0/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_1/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_1/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_1/residual_0/Reshape/shapePack(decoder/layer_1/residual_0/strided_slice*decoder/layer_1/residual_0/Reshape/shape/1*decoder/layer_1/residual_0/Reshape/shape/2*decoder/layer_1/residual_0/Reshape/shape/3*decoder/layer_1/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_1/residual_0/ReshapeReshape"decoder/layer_1/residual_0/BiasAdd(decoder/layer_1/residual_0/Reshape/shape*<
_output_shapes*
(:&                  *
T0
b
 decoder/layer_1/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_1/residual_0/split/split_dimConst*
dtype0*
_output_shapes
: *
valueB :
         
щ
 decoder/layer_1/residual_0/splitSplit*decoder/layer_1/residual_0/split/split_dim"decoder/layer_1/residual_0/Reshape*
	num_split*d
_output_shapesR
P:&                  :&                  *
T0
ј
decoder/layer_1/residual_0/EluElu decoder/layer_1/residual_0/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_1/residual_0/NegNeg"decoder/layer_1/residual_0/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_1/residual_0/Elu_1Eludecoder/layer_1/residual_0/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_1/residual_0/Neg_1Neg decoder/layer_1/residual_0/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_1/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_1/residual_0/concatConcatV2decoder/layer_1/residual_0/Elu decoder/layer_1/residual_0/Neg_1&decoder/layer_1/residual_0/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
n
,decoder/layer_1/residual_0/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_1/residual_0/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
,decoder/layer_1/residual_0/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_1/residual_0/Reshape_1/shapePack(decoder/layer_1/residual_0/strided_slice,decoder/layer_1/residual_0/Reshape_1/shape/1,decoder/layer_1/residual_0/Reshape_1/shape/2,decoder/layer_1/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_1/residual_0/Reshape_1Reshape!decoder/layer_1/residual_0/concat*decoder/layer_1/residual_0/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*
valueB
 *лвл╝*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*
valueB
 *лвл<*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
╩
Jdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_1/residual_1/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_1/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ*<
shared_name-+decoder/layer_1/residual_1/depthwise_kernel
Д
Ldecoder/layer_1/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_1/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_1/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_1/residual_1/depthwise_kernelFdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_1/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_1/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*
valueB
 *  ђй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*
valueB
 *  ђ=*
dtype0*
_output_shapes
: 
│
Tdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*(
_output_shapes
:ђђ
п
Fdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_1/residual_1/pointwise_kernelVarHandleOp*<
shared_name-+decoder/layer_1/residual_1/pointwise_kernel*>
_class4
20loc:@decoder/layer_1/residual_1/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:ђђ
Д
Ldecoder/layer_1/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_1/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_1/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_1/residual_1/pointwise_kernelFdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_1/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_1/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_1/residual_1/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_1/residual_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_1/residual_1/biasVarHandleOp*2
_class(
&$loc:@decoder/layer_1/residual_1/bias*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_1/residual_1/bias
Ј
@decoder/layer_1/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_1/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_1/residual_1/bias/AssignAssignVariableOpdecoder/layer_1/residual_1/bias1decoder/layer_1/residual_1/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_1/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_1/residual_1/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_1/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_1/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_1/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_1/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_1/residual_1/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_1/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Њ
5decoder/layer_1/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_1/residual_0/Reshape_1:decoder/layer_1/residual_1/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ј
+decoder/layer_1/residual_1/separable_conv2dConv2D5decoder/layer_1/residual_1/separable_conv2d/depthwise<decoder/layer_1/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*0
_output_shapes
:         ђ
ј
1decoder/layer_1/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_1/residual_1/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_1/residual_1/BiasAddBiasAdd+decoder/layer_1/residual_1/separable_conv2d1decoder/layer_1/residual_1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
r
 decoder/layer_1/residual_1/ShapeShape"decoder/layer_1/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_1/residual_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
z
0decoder/layer_1/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_1/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_1/residual_1/strided_sliceStridedSlice decoder/layer_1/residual_1/Shape.decoder/layer_1/residual_1/strided_slice/stack0decoder/layer_1/residual_1/strided_slice/stack_10decoder/layer_1/residual_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
l
*decoder/layer_1/residual_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_1/residual_1/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_1/residual_1/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_1/residual_1/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_1/residual_1/Reshape/shapePack(decoder/layer_1/residual_1/strided_slice*decoder/layer_1/residual_1/Reshape/shape/1*decoder/layer_1/residual_1/Reshape/shape/2*decoder/layer_1/residual_1/Reshape/shape/3*decoder/layer_1/residual_1/Reshape/shape/4*
N*
_output_shapes
:*
T0
┬
"decoder/layer_1/residual_1/ReshapeReshape"decoder/layer_1/residual_1/BiasAdd(decoder/layer_1/residual_1/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_1/residual_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_1/residual_1/split/split_dimConst*
dtype0*
_output_shapes
: *
valueB :
         
щ
 decoder/layer_1/residual_1/splitSplit*decoder/layer_1/residual_1/split/split_dim"decoder/layer_1/residual_1/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ј
decoder/layer_1/residual_1/EluElu decoder/layer_1/residual_1/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_1/residual_1/NegNeg"decoder/layer_1/residual_1/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_1/residual_1/Elu_1Eludecoder/layer_1/residual_1/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_1/residual_1/Neg_1Neg decoder/layer_1/residual_1/Elu_1*<
_output_shapes*
(:&                  *
T0
q
&decoder/layer_1/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_1/residual_1/concatConcatV2decoder/layer_1/residual_1/Elu decoder/layer_1/residual_1/Neg_1&decoder/layer_1/residual_1/concat/axis*
N*<
_output_shapes*
(:&                  *
T0
n
,decoder/layer_1/residual_1/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_1/residual_1/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
,decoder/layer_1/residual_1/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value
B :ђ
ћ
*decoder/layer_1/residual_1/Reshape_1/shapePack(decoder/layer_1/residual_1/strided_slice,decoder/layer_1/residual_1/Reshape_1/shape/1,decoder/layer_1/residual_1/Reshape_1/shape/2,decoder/layer_1/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_1/residual_1/Reshape_1Reshape!decoder/layer_1/residual_1/concat*decoder/layer_1/residual_1/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
а
decoder/layer_1/addAddV2!decoder/layer_1/strided/Reshape_1$decoder/layer_1/residual_1/Reshape_1*
T0*0
_output_shapes
:         ђ
X
decoder/layer_1/ShapeShapedecoder/layer_1/add*
T0*
_output_shapes
:
m
#decoder/layer_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
§
decoder/layer_1/strided_sliceStridedSlicedecoder/layer_1/Shape#decoder/layer_1/strided_slice/stack%decoder/layer_1/strided_slice/stack_1%decoder/layer_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
╗
4decoder/layer_1/ln/layer_norm_scale/Initializer/onesConst*
dtype0*
_output_shapes	
:ђ*6
_class,
*(loc:@decoder/layer_1/ln/layer_norm_scale*
valueBђ*  ђ?
О
#decoder/layer_1/ln/layer_norm_scaleVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*4
shared_name%#decoder/layer_1/ln/layer_norm_scale*6
_class,
*(loc:@decoder/layer_1/ln/layer_norm_scale
Ќ
Ddecoder/layer_1/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_1/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_1/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_1/ln/layer_norm_scale4decoder/layer_1/ln/layer_norm_scale/Initializer/ones*
dtype0
ў
7decoder/layer_1/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_1/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
║
4decoder/layer_1/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_1/ln/layer_norm_bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
н
"decoder/layer_1/ln/layer_norm_biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*3
shared_name$"decoder/layer_1/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_1/ln/layer_norm_bias
Ћ
Cdecoder/layer_1/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_1/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_1/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_1/ln/layer_norm_bias4decoder/layer_1/ln/layer_norm_bias/Initializer/zeros*
dtype0
ќ
6decoder/layer_1/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_1/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
]
decoder/layer_1/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
ѓ
!decoder/layer_1/ln/ReadVariableOpReadVariableOp#decoder/layer_1/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
Ѓ
#decoder/layer_1/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_1/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
|
)decoder/layer_1/ln/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
         
ф
decoder/layer_1/ln/MeanMeandecoder/layer_1/add)decoder/layer_1/ln/Mean/reduction_indices*/
_output_shapes
:         *
	keep_dims(*
T0
б
$decoder/layer_1/ln/SquaredDifferenceSquaredDifferencedecoder/layer_1/adddecoder/layer_1/ln/Mean*
T0*0
_output_shapes
:         ђ
~
+decoder/layer_1/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┐
decoder/layer_1/ln/Mean_1Mean$decoder/layer_1/ln/SquaredDifference+decoder/layer_1/ln/Mean_1/reduction_indices*
	keep_dims(*
T0*/
_output_shapes
:         
є
decoder/layer_1/ln/subSubdecoder/layer_1/adddecoder/layer_1/ln/Mean*
T0*0
_output_shapes
:         ђ
ј
decoder/layer_1/ln/addAddV2decoder/layer_1/ln/Mean_1decoder/layer_1/ln/Const*/
_output_shapes
:         *
T0
s
decoder/layer_1/ln/RsqrtRsqrtdecoder/layer_1/ln/add*
T0*/
_output_shapes
:         
і
decoder/layer_1/ln/mulMuldecoder/layer_1/ln/subdecoder/layer_1/ln/Rsqrt*
T0*0
_output_shapes
:         ђ
Ћ
decoder/layer_1/ln/mul_1Muldecoder/layer_1/ln/mul!decoder/layer_1/ln/ReadVariableOp*
T0*0
_output_shapes
:         ђ
Џ
decoder/layer_1/ln/add_1AddV2decoder/layer_1/ln/mul_1#decoder/layer_1/ln/ReadVariableOp_1*0
_output_shapes
:         ђ*
T0
╦
?decoder/layer_2/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_2/strided/kernel*%
valueB"            *
dtype0*
_output_shapes
:
х
=decoder/layer_2/strided/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_2/strided/kernel*
valueB
 *зх╝
х
=decoder/layer_2/strided/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@decoder/layer_2/strided/kernel*
valueB
 *зх<*
dtype0*
_output_shapes
: 
ї
Gdecoder/layer_2/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_2/strided/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:ђђ*
T0*1
_class'
%#loc:@decoder/layer_2/strided/kernel
ќ
=decoder/layer_2/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_2/strided/kernel/Initializer/random_uniform/max=decoder/layer_2/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_2/strided/kernel*
_output_shapes
: 
▓
=decoder/layer_2/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_2/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_2/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_2/strided/kernel*(
_output_shapes
:ђђ
ц
9decoder/layer_2/strided/kernel/Initializer/random_uniformAdd=decoder/layer_2/strided/kernel/Initializer/random_uniform/mul=decoder/layer_2/strided/kernel/Initializer/random_uniform/min*(
_output_shapes
:ђђ*
T0*1
_class'
%#loc:@decoder/layer_2/strided/kernel
Н
decoder/layer_2/strided/kernelVarHandleOp*/
shared_name decoder/layer_2/strided/kernel*1
_class'
%#loc:@decoder/layer_2/strided/kernel*
dtype0*
_output_shapes
: *
shape:ђђ
Ї
?decoder/layer_2/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_2/strided/kernel*
_output_shapes
: 
А
%decoder/layer_2/strided/kernel/AssignAssignVariableOpdecoder/layer_2/strided/kernel9decoder/layer_2/strided/kernel/Initializer/random_uniform*
dtype0
Џ
2decoder/layer_2/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_2/strided/kernel*
dtype0*(
_output_shapes
:ђђ
«
.decoder/layer_2/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_2/strided/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
┬
decoder/layer_2/strided/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*-
shared_namedecoder/layer_2/strided/bias*/
_class%
#!loc:@decoder/layer_2/strided/bias
Ѕ
=decoder/layer_2/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_2/strided/bias*
_output_shapes
: 
њ
#decoder/layer_2/strided/bias/AssignAssignVariableOpdecoder/layer_2/strided/bias.decoder/layer_2/strided/bias/Initializer/zeros*
dtype0
і
0decoder/layer_2/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_2/strided/bias*
dtype0*
_output_shapes	
:ђ
e
decoder/layer_2/strided/ShapeShapedecoder/layer_1/ln/add_1*
T0*
_output_shapes
:
u
+decoder/layer_2/strided/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-decoder/layer_2/strided/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-decoder/layer_2/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_2/strided/strided_sliceStridedSlicedecoder/layer_2/strided/Shape+decoder/layer_2/strided/strided_slice/stack-decoder/layer_2/strided/strided_slice/stack_1-decoder/layer_2/strided/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_2/strided/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_2/strided/strided_slice_1StridedSlicedecoder/layer_2/strided/Shape-decoder/layer_2/strided/strided_slice_1/stack/decoder/layer_2/strided/strided_slice_1/stack_1/decoder/layer_2/strided/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_2/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_2/strided/strided_slice_2StridedSlicedecoder/layer_2/strided/Shape-decoder/layer_2/strided/strided_slice_2/stack/decoder/layer_2/strided/strided_slice_2/stack_1/decoder/layer_2/strided/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
_
decoder/layer_2/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_2/strided/mulMul'decoder/layer_2/strided/strided_slice_1decoder/layer_2/strided/mul/y*
_output_shapes
: *
T0
a
decoder/layer_2/strided/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
decoder/layer_2/strided/mul_1Mul'decoder/layer_2/strided/strided_slice_2decoder/layer_2/strided/mul_1/y*
_output_shapes
: *
T0
b
decoder/layer_2/strided/stack/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
О
decoder/layer_2/strided/stackPack%decoder/layer_2/strided/strided_slicedecoder/layer_2/strided/muldecoder/layer_2/strided/mul_1decoder/layer_2/strided/stack/3*
T0*
N*
_output_shapes
:
w
-decoder/layer_2/strided/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/decoder/layer_2/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_2/strided/strided_slice_3StridedSlicedecoder/layer_2/strided/stack-decoder/layer_2/strided/strided_slice_3/stack/decoder/layer_2/strided/strided_slice_3/stack_1/decoder/layer_2/strided/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
а
7decoder/layer_2/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_2/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ћ
(decoder/layer_2/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_2/strided/stack7decoder/layer_2/strided/conv2d_transpose/ReadVariableOpdecoder/layer_1/ln/add_1*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ѕ
.decoder/layer_2/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_2/strided/bias*
dtype0*
_output_shapes	
:ђ
┐
decoder/layer_2/strided/BiasAddBiasAdd(decoder/layer_2/strided/conv2d_transpose.decoder/layer_2/strided/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
n
decoder/layer_2/strided/Shape_1Shapedecoder/layer_2/strided/BiasAdd*
_output_shapes
:*
T0
w
-decoder/layer_2/strided/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_2/strided/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
»
'decoder/layer_2/strided/strided_slice_4StridedSlicedecoder/layer_2/strided/Shape_1-decoder/layer_2/strided/strided_slice_4/stack/decoder/layer_2/strided/strided_slice_4/stack_1/decoder/layer_2/strided/strided_slice_4/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
i
'decoder/layer_2/strided/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
i
'decoder/layer_2/strided/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_2/strided/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
i
'decoder/layer_2/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_2/strided/Reshape/shapePack'decoder/layer_2/strided/strided_slice_4'decoder/layer_2/strided/Reshape/shape/1'decoder/layer_2/strided/Reshape/shape/2'decoder/layer_2/strided/Reshape/shape/3'decoder/layer_2/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_2/strided/ReshapeReshapedecoder/layer_2/strided/BiasAdd%decoder/layer_2/strided/Reshape/shape*
T0*<
_output_shapes*
(:&                  
_
decoder/layer_2/strided/ConstConst*
dtype0*
_output_shapes
: *
value	B :
r
'decoder/layer_2/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_2/strided/splitSplit'decoder/layer_2/strided/split/split_dimdecoder/layer_2/strided/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ѕ
decoder/layer_2/strided/EluEludecoder/layer_2/strided/split*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_2/strided/NegNegdecoder/layer_2/strided/split:1*
T0*<
_output_shapes*
(:&                  
ѕ
decoder/layer_2/strided/Elu_1Eludecoder/layer_2/strided/Neg*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_2/strided/Neg_1Negdecoder/layer_2/strided/Elu_1*<
_output_shapes*
(:&                  *
T0
n
#decoder/layer_2/strided/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
decoder/layer_2/strided/concatConcatV2decoder/layer_2/strided/Eludecoder/layer_2/strided/Neg_1#decoder/layer_2/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
k
)decoder/layer_2/strided/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
k
)decoder/layer_2/strided/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
l
)decoder/layer_2/strided/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
Є
'decoder/layer_2/strided/Reshape_1/shapePack'decoder/layer_2/strided/strided_slice_4)decoder/layer_2/strided/Reshape_1/shape/1)decoder/layer_2/strided/Reshape_1/shape/2)decoder/layer_2/strided/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
░
!decoder/layer_2/strided/Reshape_1Reshapedecoder/layer_2/strided/concat'decoder/layer_2/strided/Reshape_1/shape*0
_output_shapes
:         ђ*
T0
т
Ldecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel*
valueB
 *иЮPй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel*
valueB
 *иЮP=*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel
╩
Jdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel
О
Fdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_2/residual_0/depthwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*<
shared_name-+decoder/layer_2/residual_0/depthwise_kernel*>
_class4
20loc:@decoder/layer_2/residual_0/depthwise_kernel
Д
Ldecoder/layer_2/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_2/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_2/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_2/residual_0/depthwise_kernelFdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_2/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_2/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*
valueB
 *зх=
│
Tdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
п
Fdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_2/residual_0/pointwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_2/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:ђђ*<
shared_name-+decoder/layer_2/residual_0/pointwise_kernel
Д
Ldecoder/layer_2/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_2/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_2/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_2/residual_0/pointwise_kernelFdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_2/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_2/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_2/residual_0/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*2
_class(
&$loc:@decoder/layer_2/residual_0/bias*
valueBђ*    
╦
decoder/layer_2/residual_0/biasVarHandleOp*2
_class(
&$loc:@decoder/layer_2/residual_0/bias*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_2/residual_0/bias
Ј
@decoder/layer_2/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_2/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_2/residual_0/bias/AssignAssignVariableOpdecoder/layer_2/residual_0/bias1decoder/layer_2/residual_0/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_2/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_2/residual_0/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_2/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_2/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_2/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_2/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_2/residual_0/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_2/residual_0/separable_conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
љ
5decoder/layer_2/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_2/strided/Reshape_1:decoder/layer_2/residual_0/separable_conv2d/ReadVariableOp*
T0*
strides
*0
_output_shapes
:         ђ*
paddingSAME
ј
+decoder/layer_2/residual_0/separable_conv2dConv2D5decoder/layer_2/residual_0/separable_conv2d/depthwise<decoder/layer_2/residual_0/separable_conv2d/ReadVariableOp_1*
T0*
strides
*0
_output_shapes
:         ђ*
paddingVALID
ј
1decoder/layer_2/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_2/residual_0/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_2/residual_0/BiasAddBiasAdd+decoder/layer_2/residual_0/separable_conv2d1decoder/layer_2/residual_0/BiasAdd/ReadVariableOp*0
_output_shapes
:         ђ*
T0
r
 decoder/layer_2/residual_0/ShapeShape"decoder/layer_2/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_2/residual_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_2/residual_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_2/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_2/residual_0/strided_sliceStridedSlice decoder/layer_2/residual_0/Shape.decoder/layer_2/residual_0/strided_slice/stack0decoder/layer_2/residual_0/strided_slice/stack_10decoder/layer_2/residual_0/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
l
*decoder/layer_2/residual_0/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_2/residual_0/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_2/residual_0/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
valueB :
         
l
*decoder/layer_2/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_2/residual_0/Reshape/shapePack(decoder/layer_2/residual_0/strided_slice*decoder/layer_2/residual_0/Reshape/shape/1*decoder/layer_2/residual_0/Reshape/shape/2*decoder/layer_2/residual_0/Reshape/shape/3*decoder/layer_2/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_2/residual_0/ReshapeReshape"decoder/layer_2/residual_0/BiasAdd(decoder/layer_2/residual_0/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_2/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_2/residual_0/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_2/residual_0/splitSplit*decoder/layer_2/residual_0/split/split_dim"decoder/layer_2/residual_0/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ј
decoder/layer_2/residual_0/EluElu decoder/layer_2/residual_0/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_2/residual_0/NegNeg"decoder/layer_2/residual_0/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_2/residual_0/Elu_1Eludecoder/layer_2/residual_0/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_2/residual_0/Neg_1Neg decoder/layer_2/residual_0/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_2/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_2/residual_0/concatConcatV2decoder/layer_2/residual_0/Elu decoder/layer_2/residual_0/Neg_1&decoder/layer_2/residual_0/concat/axis*
N*<
_output_shapes*
(:&                  *
T0
n
,decoder/layer_2/residual_0/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
n
,decoder/layer_2/residual_0/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
,decoder/layer_2/residual_0/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_2/residual_0/Reshape_1/shapePack(decoder/layer_2/residual_0/strided_slice,decoder/layer_2/residual_0/Reshape_1/shape/1,decoder/layer_2/residual_0/Reshape_1/shape/2,decoder/layer_2/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_2/residual_0/Reshape_1Reshape!decoder/layer_2/residual_0/concat*decoder/layer_2/residual_0/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*
valueB
 *Uей*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*
valueB
 *Uе=
▓
Tdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
╩
Jdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel
О
Fdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_2/residual_1/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_2/residual_1/depthwise_kernel*>
_class4
20loc:@decoder/layer_2/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
Д
Ldecoder/layer_2/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_2/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_2/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_2/residual_1/depthwise_kernelFdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_2/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_2/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*
valueB
 *зх=
│
Tdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*(
_output_shapes
:ђђ
п
Fdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform/min*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel
Ч
+decoder/layer_2/residual_1/pointwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_2/residual_1/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:ђђ*<
shared_name-+decoder/layer_2/residual_1/pointwise_kernel
Д
Ldecoder/layer_2/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_2/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_2/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_2/residual_1/pointwise_kernelFdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_2/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_2/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_2/residual_1/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_2/residual_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_2/residual_1/biasVarHandleOp*0
shared_name!decoder/layer_2/residual_1/bias*2
_class(
&$loc:@decoder/layer_2/residual_1/bias*
dtype0*
_output_shapes
: *
shape:ђ
Ј
@decoder/layer_2/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_2/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_2/residual_1/bias/AssignAssignVariableOpdecoder/layer_2/residual_1/bias1decoder/layer_2/residual_1/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_2/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_2/residual_1/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_2/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_2/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_2/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_2/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_2/residual_1/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_2/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Њ
5decoder/layer_2/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_2/residual_0/Reshape_1:decoder/layer_2/residual_1/separable_conv2d/ReadVariableOp*
strides
*0
_output_shapes
:         ђ*
paddingSAME*
T0
ј
+decoder/layer_2/residual_1/separable_conv2dConv2D5decoder/layer_2/residual_1/separable_conv2d/depthwise<decoder/layer_2/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*0
_output_shapes
:         ђ
ј
1decoder/layer_2/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_2/residual_1/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_2/residual_1/BiasAddBiasAdd+decoder/layer_2/residual_1/separable_conv2d1decoder/layer_2/residual_1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
r
 decoder/layer_2/residual_1/ShapeShape"decoder/layer_2/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_2/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_2/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_2/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_2/residual_1/strided_sliceStridedSlice decoder/layer_2/residual_1/Shape.decoder/layer_2/residual_1/strided_slice/stack0decoder/layer_2/residual_1/strided_slice/stack_10decoder/layer_2/residual_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
l
*decoder/layer_2/residual_1/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
l
*decoder/layer_2/residual_1/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_2/residual_1/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
valueB :
         
l
*decoder/layer_2/residual_1/Reshape/shape/4Const*
dtype0*
_output_shapes
: *
value	B :
И
(decoder/layer_2/residual_1/Reshape/shapePack(decoder/layer_2/residual_1/strided_slice*decoder/layer_2/residual_1/Reshape/shape/1*decoder/layer_2/residual_1/Reshape/shape/2*decoder/layer_2/residual_1/Reshape/shape/3*decoder/layer_2/residual_1/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_2/residual_1/ReshapeReshape"decoder/layer_2/residual_1/BiasAdd(decoder/layer_2/residual_1/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_2/residual_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_2/residual_1/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_2/residual_1/splitSplit*decoder/layer_2/residual_1/split/split_dim"decoder/layer_2/residual_1/Reshape*
	num_split*d
_output_shapesR
P:&                  :&                  *
T0
ј
decoder/layer_2/residual_1/EluElu decoder/layer_2/residual_1/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_2/residual_1/NegNeg"decoder/layer_2/residual_1/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_2/residual_1/Elu_1Eludecoder/layer_2/residual_1/Neg*<
_output_shapes*
(:&                  *
T0
љ
 decoder/layer_2/residual_1/Neg_1Neg decoder/layer_2/residual_1/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_2/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_2/residual_1/concatConcatV2decoder/layer_2/residual_1/Elu decoder/layer_2/residual_1/Neg_1&decoder/layer_2/residual_1/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
n
,decoder/layer_2/residual_1/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
n
,decoder/layer_2/residual_1/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
,decoder/layer_2/residual_1/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_2/residual_1/Reshape_1/shapePack(decoder/layer_2/residual_1/strided_slice,decoder/layer_2/residual_1/Reshape_1/shape/1,decoder/layer_2/residual_1/Reshape_1/shape/2,decoder/layer_2/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_2/residual_1/Reshape_1Reshape!decoder/layer_2/residual_1/concat*decoder/layer_2/residual_1/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
а
decoder/layer_2/addAddV2!decoder/layer_2/strided/Reshape_1$decoder/layer_2/residual_1/Reshape_1*
T0*0
_output_shapes
:         ђ
X
decoder/layer_2/ShapeShapedecoder/layer_2/add*
T0*
_output_shapes
:
m
#decoder/layer_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_2/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
§
decoder/layer_2/strided_sliceStridedSlicedecoder/layer_2/Shape#decoder/layer_2/strided_slice/stack%decoder/layer_2/strided_slice/stack_1%decoder/layer_2/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
╗
4decoder/layer_2/ln/layer_norm_scale/Initializer/onesConst*
dtype0*
_output_shapes	
:ђ*6
_class,
*(loc:@decoder/layer_2/ln/layer_norm_scale*
valueBђ*  ђ?
О
#decoder/layer_2/ln/layer_norm_scaleVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*4
shared_name%#decoder/layer_2/ln/layer_norm_scale*6
_class,
*(loc:@decoder/layer_2/ln/layer_norm_scale
Ќ
Ddecoder/layer_2/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_2/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_2/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_2/ln/layer_norm_scale4decoder/layer_2/ln/layer_norm_scale/Initializer/ones*
dtype0
ў
7decoder/layer_2/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_2/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
║
4decoder/layer_2/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_2/ln/layer_norm_bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
н
"decoder/layer_2/ln/layer_norm_biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*3
shared_name$"decoder/layer_2/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_2/ln/layer_norm_bias
Ћ
Cdecoder/layer_2/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_2/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_2/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_2/ln/layer_norm_bias4decoder/layer_2/ln/layer_norm_bias/Initializer/zeros*
dtype0
ќ
6decoder/layer_2/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_2/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
]
decoder/layer_2/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
ѓ
!decoder/layer_2/ln/ReadVariableOpReadVariableOp#decoder/layer_2/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
Ѓ
#decoder/layer_2/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_2/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
|
)decoder/layer_2/ln/Mean/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
ф
decoder/layer_2/ln/MeanMeandecoder/layer_2/add)decoder/layer_2/ln/Mean/reduction_indices*
T0*/
_output_shapes
:         *
	keep_dims(
б
$decoder/layer_2/ln/SquaredDifferenceSquaredDifferencedecoder/layer_2/adddecoder/layer_2/ln/Mean*
T0*0
_output_shapes
:         ђ
~
+decoder/layer_2/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┐
decoder/layer_2/ln/Mean_1Mean$decoder/layer_2/ln/SquaredDifference+decoder/layer_2/ln/Mean_1/reduction_indices*
T0*/
_output_shapes
:         *
	keep_dims(
є
decoder/layer_2/ln/subSubdecoder/layer_2/adddecoder/layer_2/ln/Mean*
T0*0
_output_shapes
:         ђ
ј
decoder/layer_2/ln/addAddV2decoder/layer_2/ln/Mean_1decoder/layer_2/ln/Const*/
_output_shapes
:         *
T0
s
decoder/layer_2/ln/RsqrtRsqrtdecoder/layer_2/ln/add*/
_output_shapes
:         *
T0
і
decoder/layer_2/ln/mulMuldecoder/layer_2/ln/subdecoder/layer_2/ln/Rsqrt*
T0*0
_output_shapes
:         ђ
Ћ
decoder/layer_2/ln/mul_1Muldecoder/layer_2/ln/mul!decoder/layer_2/ln/ReadVariableOp*
T0*0
_output_shapes
:         ђ
Џ
decoder/layer_2/ln/add_1AddV2decoder/layer_2/ln/mul_1#decoder/layer_2/ln/ReadVariableOp_1*0
_output_shapes
:         ђ*
T0
╦
?decoder/layer_3/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_3/strided/kernel*%
valueB"      ђ      *
dtype0*
_output_shapes
:
х
=decoder/layer_3/strided/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_3/strided/kernel*
valueB
 *   й
х
=decoder/layer_3/strided/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@decoder/layer_3/strided/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ї
Gdecoder/layer_3/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_3/strided/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@decoder/layer_3/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ќ
=decoder/layer_3/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_3/strided/kernel/Initializer/random_uniform/max=decoder/layer_3/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_3/strided/kernel*
_output_shapes
: 
▓
=decoder/layer_3/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_3/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_3/strided/kernel/Initializer/random_uniform/sub*(
_output_shapes
:ђђ*
T0*1
_class'
%#loc:@decoder/layer_3/strided/kernel
ц
9decoder/layer_3/strided/kernel/Initializer/random_uniformAdd=decoder/layer_3/strided/kernel/Initializer/random_uniform/mul=decoder/layer_3/strided/kernel/Initializer/random_uniform/min*(
_output_shapes
:ђђ*
T0*1
_class'
%#loc:@decoder/layer_3/strided/kernel
Н
decoder/layer_3/strided/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђђ*/
shared_name decoder/layer_3/strided/kernel*1
_class'
%#loc:@decoder/layer_3/strided/kernel
Ї
?decoder/layer_3/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_3/strided/kernel*
_output_shapes
: 
А
%decoder/layer_3/strided/kernel/AssignAssignVariableOpdecoder/layer_3/strided/kernel9decoder/layer_3/strided/kernel/Initializer/random_uniform*
dtype0
Џ
2decoder/layer_3/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_3/strided/kernel*
dtype0*(
_output_shapes
:ђђ
«
.decoder/layer_3/strided/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*/
_class%
#!loc:@decoder/layer_3/strided/bias*
valueBђ*    
┬
decoder/layer_3/strided/biasVarHandleOp*
shape:ђ*-
shared_namedecoder/layer_3/strided/bias*/
_class%
#!loc:@decoder/layer_3/strided/bias*
dtype0*
_output_shapes
: 
Ѕ
=decoder/layer_3/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_3/strided/bias*
_output_shapes
: 
њ
#decoder/layer_3/strided/bias/AssignAssignVariableOpdecoder/layer_3/strided/bias.decoder/layer_3/strided/bias/Initializer/zeros*
dtype0
і
0decoder/layer_3/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_3/strided/bias*
dtype0*
_output_shapes	
:ђ
e
decoder/layer_3/strided/ShapeShapedecoder/layer_2/ln/add_1*
T0*
_output_shapes
:
u
+decoder/layer_3/strided/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-decoder/layer_3/strided/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-decoder/layer_3/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_3/strided/strided_sliceStridedSlicedecoder/layer_3/strided/Shape+decoder/layer_3/strided/strided_slice/stack-decoder/layer_3/strided/strided_slice/stack_1-decoder/layer_3/strided/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
w
-decoder/layer_3/strided/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_3/strided/strided_slice_1StridedSlicedecoder/layer_3/strided/Shape-decoder/layer_3/strided/strided_slice_1/stack/decoder/layer_3/strided/strided_slice_1/stack_1/decoder/layer_3/strided/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
w
-decoder/layer_3/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_3/strided/strided_slice_2StridedSlicedecoder/layer_3/strided/Shape-decoder/layer_3/strided/strided_slice_2/stack/decoder/layer_3/strided/strided_slice_2/stack_1/decoder/layer_3/strided/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
_
decoder/layer_3/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_3/strided/mulMul'decoder/layer_3/strided/strided_slice_1decoder/layer_3/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_3/strided/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Ј
decoder/layer_3/strided/mul_1Mul'decoder/layer_3/strided/strided_slice_2decoder/layer_3/strided/mul_1/y*
T0*
_output_shapes
: 
b
decoder/layer_3/strided/stack/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
О
decoder/layer_3/strided/stackPack%decoder/layer_3/strided/strided_slicedecoder/layer_3/strided/muldecoder/layer_3/strided/mul_1decoder/layer_3/strided/stack/3*
T0*
N*
_output_shapes
:
w
-decoder/layer_3/strided/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_3/strided/strided_slice_3StridedSlicedecoder/layer_3/strided/stack-decoder/layer_3/strided/strided_slice_3/stack/decoder/layer_3/strided/strided_slice_3/stack_1/decoder/layer_3/strided/strided_slice_3/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
а
7decoder/layer_3/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_3/strided/kernel*
dtype0*(
_output_shapes
:ђђ
ћ
(decoder/layer_3/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_3/strided/stack7decoder/layer_3/strided/conv2d_transpose/ReadVariableOpdecoder/layer_2/ln/add_1*
strides
*0
_output_shapes
:         ђ*
paddingSAME*
T0
ѕ
.decoder/layer_3/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_3/strided/bias*
dtype0*
_output_shapes	
:ђ
┐
decoder/layer_3/strided/BiasAddBiasAdd(decoder/layer_3/strided/conv2d_transpose.decoder/layer_3/strided/BiasAdd/ReadVariableOp*0
_output_shapes
:         ђ*
T0
n
decoder/layer_3/strided/Shape_1Shapedecoder/layer_3/strided/BiasAdd*
T0*
_output_shapes
:
w
-decoder/layer_3/strided/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_3/strided/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
»
'decoder/layer_3/strided/strided_slice_4StridedSlicedecoder/layer_3/strided/Shape_1-decoder/layer_3/strided/strided_slice_4/stack/decoder/layer_3/strided/strided_slice_4/stack_1/decoder/layer_3/strided/strided_slice_4/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
i
'decoder/layer_3/strided/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
i
'decoder/layer_3/strided/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
r
'decoder/layer_3/strided/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
valueB :
         
i
'decoder/layer_3/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_3/strided/Reshape/shapePack'decoder/layer_3/strided/strided_slice_4'decoder/layer_3/strided/Reshape/shape/1'decoder/layer_3/strided/Reshape/shape/2'decoder/layer_3/strided/Reshape/shape/3'decoder/layer_3/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_3/strided/ReshapeReshapedecoder/layer_3/strided/BiasAdd%decoder/layer_3/strided/Reshape/shape*
T0*<
_output_shapes*
(:&                  
_
decoder/layer_3/strided/ConstConst*
dtype0*
_output_shapes
: *
value	B :
r
'decoder/layer_3/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_3/strided/splitSplit'decoder/layer_3/strided/split/split_dimdecoder/layer_3/strided/Reshape*
	num_split*d
_output_shapesR
P:&                  :&                  *
T0
ѕ
decoder/layer_3/strided/EluEludecoder/layer_3/strided/split*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_3/strided/NegNegdecoder/layer_3/strided/split:1*
T0*<
_output_shapes*
(:&                  
ѕ
decoder/layer_3/strided/Elu_1Eludecoder/layer_3/strided/Neg*
T0*<
_output_shapes*
(:&                  
і
decoder/layer_3/strided/Neg_1Negdecoder/layer_3/strided/Elu_1*
T0*<
_output_shapes*
(:&                  
n
#decoder/layer_3/strided/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
█
decoder/layer_3/strided/concatConcatV2decoder/layer_3/strided/Eludecoder/layer_3/strided/Neg_1#decoder/layer_3/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
k
)decoder/layer_3/strided/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
k
)decoder/layer_3/strided/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
l
)decoder/layer_3/strided/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
Є
'decoder/layer_3/strided/Reshape_1/shapePack'decoder/layer_3/strided/strided_slice_4)decoder/layer_3/strided/Reshape_1/shape/1)decoder/layer_3/strided/Reshape_1/shape/2)decoder/layer_3/strided/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
░
!decoder/layer_3/strided/Reshape_1Reshapedecoder/layer_3/strided/concat'decoder/layer_3/strided/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*%
valueB"      ђ      *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*
valueB
 *I:Њй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*
valueB
 *I:Њ=*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel
╩
Jdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel
т
Jdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_3/residual_0/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_3/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ*<
shared_name-+decoder/layer_3/residual_0/depthwise_kernel
Д
Ldecoder/layer_3/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_3/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_3/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_3/residual_0/depthwise_kernelFdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_3/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_3/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*%
valueB"      ђ      *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*
valueB
 *   Й*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
│
Tdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel
Т
Jdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
п
Fdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_3/residual_0/pointwise_kernelVarHandleOp*
shape:ђђ*<
shared_name-+decoder/layer_3/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_3/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: 
Д
Ldecoder/layer_3/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_3/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_3/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_3/residual_0/pointwise_kernelFdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_3/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_3/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_3/residual_0/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_3/residual_0/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_3/residual_0/biasVarHandleOp*0
shared_name!decoder/layer_3/residual_0/bias*2
_class(
&$loc:@decoder/layer_3/residual_0/bias*
dtype0*
_output_shapes
: *
shape:ђ
Ј
@decoder/layer_3/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_3/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_3/residual_0/bias/AssignAssignVariableOpdecoder/layer_3/residual_0/bias1decoder/layer_3/residual_0/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_3/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_3/residual_0/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_3/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_3/residual_0/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_3/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_3/residual_0/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_3/residual_0/separable_conv2d/ShapeConst*%
valueB"      ђ      *
dtype0*
_output_shapes
:
і
9decoder/layer_3/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
љ
5decoder/layer_3/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_3/strided/Reshape_1:decoder/layer_3/residual_0/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:         ђ
ј
+decoder/layer_3/residual_0/separable_conv2dConv2D5decoder/layer_3/residual_0/separable_conv2d/depthwise<decoder/layer_3/residual_0/separable_conv2d/ReadVariableOp_1*
strides
*0
_output_shapes
:         ђ*
paddingVALID*
T0
ј
1decoder/layer_3/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_3/residual_0/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_3/residual_0/BiasAddBiasAdd+decoder/layer_3/residual_0/separable_conv2d1decoder/layer_3/residual_0/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
r
 decoder/layer_3/residual_0/ShapeShape"decoder/layer_3/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_3/residual_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_3/residual_0/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/layer_3/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_3/residual_0/strided_sliceStridedSlice decoder/layer_3/residual_0/Shape.decoder/layer_3/residual_0/strided_slice/stack0decoder/layer_3/residual_0/strided_slice/stack_10decoder/layer_3/residual_0/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
l
*decoder/layer_3/residual_0/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_3/residual_0/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_3/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_3/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_3/residual_0/Reshape/shapePack(decoder/layer_3/residual_0/strided_slice*decoder/layer_3/residual_0/Reshape/shape/1*decoder/layer_3/residual_0/Reshape/shape/2*decoder/layer_3/residual_0/Reshape/shape/3*decoder/layer_3/residual_0/Reshape/shape/4*
N*
_output_shapes
:*
T0
┬
"decoder/layer_3/residual_0/ReshapeReshape"decoder/layer_3/residual_0/BiasAdd(decoder/layer_3/residual_0/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_3/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_3/residual_0/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_3/residual_0/splitSplit*decoder/layer_3/residual_0/split/split_dim"decoder/layer_3/residual_0/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ј
decoder/layer_3/residual_0/EluElu decoder/layer_3/residual_0/split*
T0*<
_output_shapes*
(:&                  
љ
decoder/layer_3/residual_0/NegNeg"decoder/layer_3/residual_0/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_3/residual_0/Elu_1Eludecoder/layer_3/residual_0/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_3/residual_0/Neg_1Neg decoder/layer_3/residual_0/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_3/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_3/residual_0/concatConcatV2decoder/layer_3/residual_0/Elu decoder/layer_3/residual_0/Neg_1&decoder/layer_3/residual_0/concat/axis*
N*<
_output_shapes*
(:&                  *
T0
n
,decoder/layer_3/residual_0/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_3/residual_0/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
o
,decoder/layer_3/residual_0/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_3/residual_0/Reshape_1/shapePack(decoder/layer_3/residual_0/strided_slice,decoder/layer_3/residual_0/Reshape_1/shape/1,decoder/layer_3/residual_0/Reshape_1/shape/2,decoder/layer_3/residual_0/Reshape_1/shape/3*
N*
_output_shapes
:*
T0
╣
$decoder/layer_3/residual_0/Reshape_1Reshape!decoder/layer_3/residual_0/concat*decoder/layer_3/residual_0/Reshape_1/shape*
T0*0
_output_shapes
:         ђ
т
Ldecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*%
valueB"            *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*
valueB
 *иЮPй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*
valueB
 *иЮP=*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel
╩
Jdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform/min*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel
ч
+decoder/layer_3/residual_1/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_3/residual_1/depthwise_kernel*>
_class4
20loc:@decoder/layer_3/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
Д
Ldecoder/layer_3/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_3/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_3/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_3/residual_1/depthwise_kernelFdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_3/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_3/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*%
valueB"         ђ   
¤
Jdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*
valueB
 *   Й*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
│
Tdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
╩
Jdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*
_output_shapes
: 
Т
Jdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/sub*(
_output_shapes
:ђђ*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel
п
Fdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*(
_output_shapes
:ђђ
Ч
+decoder/layer_3/residual_1/pointwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_3/residual_1/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:ђђ*<
shared_name-+decoder/layer_3/residual_1/pointwise_kernel
Д
Ldecoder/layer_3/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_3/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_3/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_3/residual_1/pointwise_kernelFdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
х
?decoder/layer_3/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_3/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
┤
1decoder/layer_3/residual_1/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_3/residual_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_3/residual_1/biasVarHandleOp*2
_class(
&$loc:@decoder/layer_3/residual_1/bias*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_3/residual_1/bias
Ј
@decoder/layer_3/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_3/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_3/residual_1/bias/AssignAssignVariableOpdecoder/layer_3/residual_1/bias1decoder/layer_3/residual_1/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_3/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_3/residual_1/bias*
dtype0*
_output_shapes	
:ђ
»
:decoder/layer_3/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_3/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▓
<decoder/layer_3/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_3/residual_1/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
і
1decoder/layer_3/residual_1/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_3/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Њ
5decoder/layer_3/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_3/residual_0/Reshape_1:decoder/layer_3/residual_1/separable_conv2d/ReadVariableOp*
T0*
strides
*0
_output_shapes
:         ђ*
paddingSAME
ј
+decoder/layer_3/residual_1/separable_conv2dConv2D5decoder/layer_3/residual_1/separable_conv2d/depthwise<decoder/layer_3/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*0
_output_shapes
:         ђ
ј
1decoder/layer_3/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_3/residual_1/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_3/residual_1/BiasAddBiasAdd+decoder/layer_3/residual_1/separable_conv2d1decoder/layer_3/residual_1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:         ђ
r
 decoder/layer_3/residual_1/ShapeShape"decoder/layer_3/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_3/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_3/residual_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0decoder/layer_3/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_3/residual_1/strided_sliceStridedSlice decoder/layer_3/residual_1/Shape.decoder/layer_3/residual_1/strided_slice/stack0decoder/layer_3/residual_1/strided_slice/stack_10decoder/layer_3/residual_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
l
*decoder/layer_3/residual_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*decoder/layer_3/residual_1/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_3/residual_1/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_3/residual_1/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_3/residual_1/Reshape/shapePack(decoder/layer_3/residual_1/strided_slice*decoder/layer_3/residual_1/Reshape/shape/1*decoder/layer_3/residual_1/Reshape/shape/2*decoder/layer_3/residual_1/Reshape/shape/3*decoder/layer_3/residual_1/Reshape/shape/4*
N*
_output_shapes
:*
T0
┬
"decoder/layer_3/residual_1/ReshapeReshape"decoder/layer_3/residual_1/BiasAdd(decoder/layer_3/residual_1/Reshape/shape*
T0*<
_output_shapes*
(:&                  
b
 decoder/layer_3/residual_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_3/residual_1/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_3/residual_1/splitSplit*decoder/layer_3/residual_1/split/split_dim"decoder/layer_3/residual_1/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                  :&                  
ј
decoder/layer_3/residual_1/EluElu decoder/layer_3/residual_1/split*<
_output_shapes*
(:&                  *
T0
љ
decoder/layer_3/residual_1/NegNeg"decoder/layer_3/residual_1/split:1*
T0*<
_output_shapes*
(:&                  
ј
 decoder/layer_3/residual_1/Elu_1Eludecoder/layer_3/residual_1/Neg*
T0*<
_output_shapes*
(:&                  
љ
 decoder/layer_3/residual_1/Neg_1Neg decoder/layer_3/residual_1/Elu_1*
T0*<
_output_shapes*
(:&                  
q
&decoder/layer_3/residual_1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
у
!decoder/layer_3/residual_1/concatConcatV2decoder/layer_3/residual_1/Elu decoder/layer_3/residual_1/Neg_1&decoder/layer_3/residual_1/concat/axis*
T0*
N*<
_output_shapes*
(:&                  
n
,decoder/layer_3/residual_1/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
n
,decoder/layer_3/residual_1/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
,decoder/layer_3/residual_1/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value
B :ђ
ћ
*decoder/layer_3/residual_1/Reshape_1/shapePack(decoder/layer_3/residual_1/strided_slice,decoder/layer_3/residual_1/Reshape_1/shape/1,decoder/layer_3/residual_1/Reshape_1/shape/2,decoder/layer_3/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_3/residual_1/Reshape_1Reshape!decoder/layer_3/residual_1/concat*decoder/layer_3/residual_1/Reshape_1/shape*0
_output_shapes
:         ђ*
T0
а
decoder/layer_3/addAddV2!decoder/layer_3/strided/Reshape_1$decoder/layer_3/residual_1/Reshape_1*
T0*0
_output_shapes
:         ђ
X
decoder/layer_3/ShapeShapedecoder/layer_3/add*
T0*
_output_shapes
:
m
#decoder/layer_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
decoder/layer_3/strided_sliceStridedSlicedecoder/layer_3/Shape#decoder/layer_3/strided_slice/stack%decoder/layer_3/strided_slice/stack_1%decoder/layer_3/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
╗
4decoder/layer_3/ln/layer_norm_scale/Initializer/onesConst*6
_class,
*(loc:@decoder/layer_3/ln/layer_norm_scale*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
О
#decoder/layer_3/ln/layer_norm_scaleVarHandleOp*6
_class,
*(loc:@decoder/layer_3/ln/layer_norm_scale*
dtype0*
_output_shapes
: *
shape:ђ*4
shared_name%#decoder/layer_3/ln/layer_norm_scale
Ќ
Ddecoder/layer_3/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_3/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_3/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_3/ln/layer_norm_scale4decoder/layer_3/ln/layer_norm_scale/Initializer/ones*
dtype0
ў
7decoder/layer_3/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_3/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
║
4decoder/layer_3/ln/layer_norm_bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*5
_class+
)'loc:@decoder/layer_3/ln/layer_norm_bias*
valueBђ*    
н
"decoder/layer_3/ln/layer_norm_biasVarHandleOp*
shape:ђ*3
shared_name$"decoder/layer_3/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_3/ln/layer_norm_bias*
dtype0*
_output_shapes
: 
Ћ
Cdecoder/layer_3/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_3/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_3/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_3/ln/layer_norm_bias4decoder/layer_3/ln/layer_norm_bias/Initializer/zeros*
dtype0
ќ
6decoder/layer_3/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_3/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
]
decoder/layer_3/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
ѓ
!decoder/layer_3/ln/ReadVariableOpReadVariableOp#decoder/layer_3/ln/layer_norm_scale*
dtype0*
_output_shapes	
:ђ
Ѓ
#decoder/layer_3/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_3/ln/layer_norm_bias*
dtype0*
_output_shapes	
:ђ
|
)decoder/layer_3/ln/Mean/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
ф
decoder/layer_3/ln/MeanMeandecoder/layer_3/add)decoder/layer_3/ln/Mean/reduction_indices*/
_output_shapes
:         *
	keep_dims(*
T0
б
$decoder/layer_3/ln/SquaredDifferenceSquaredDifferencedecoder/layer_3/adddecoder/layer_3/ln/Mean*
T0*0
_output_shapes
:         ђ
~
+decoder/layer_3/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┐
decoder/layer_3/ln/Mean_1Mean$decoder/layer_3/ln/SquaredDifference+decoder/layer_3/ln/Mean_1/reduction_indices*
T0*/
_output_shapes
:         *
	keep_dims(
є
decoder/layer_3/ln/subSubdecoder/layer_3/adddecoder/layer_3/ln/Mean*0
_output_shapes
:         ђ*
T0
ј
decoder/layer_3/ln/addAddV2decoder/layer_3/ln/Mean_1decoder/layer_3/ln/Const*
T0*/
_output_shapes
:         
s
decoder/layer_3/ln/RsqrtRsqrtdecoder/layer_3/ln/add*
T0*/
_output_shapes
:         
і
decoder/layer_3/ln/mulMuldecoder/layer_3/ln/subdecoder/layer_3/ln/Rsqrt*0
_output_shapes
:         ђ*
T0
Ћ
decoder/layer_3/ln/mul_1Muldecoder/layer_3/ln/mul!decoder/layer_3/ln/ReadVariableOp*
T0*0
_output_shapes
:         ђ
Џ
decoder/layer_3/ln/add_1AddV2decoder/layer_3/ln/mul_1#decoder/layer_3/ln/ReadVariableOp_1*
T0*0
_output_shapes
:         ђ
╦
?decoder/layer_4/strided/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*1
_class'
%#loc:@decoder/layer_4/strided/kernel*%
valueB"      @   ђ   
х
=decoder/layer_4/strided/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@decoder/layer_4/strided/kernel*
valueB
 *з5й*
dtype0*
_output_shapes
: 
х
=decoder/layer_4/strided/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_4/strided/kernel*
valueB
 *з5=
І
Gdecoder/layer_4/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_4/strided/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@decoder/layer_4/strided/kernel*
dtype0*'
_output_shapes
:@ђ
ќ
=decoder/layer_4/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_4/strided/kernel/Initializer/random_uniform/max=decoder/layer_4/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_4/strided/kernel*
_output_shapes
: 
▒
=decoder/layer_4/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_4/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_4/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_4/strided/kernel*'
_output_shapes
:@ђ
Б
9decoder/layer_4/strided/kernel/Initializer/random_uniformAdd=decoder/layer_4/strided/kernel/Initializer/random_uniform/mul=decoder/layer_4/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_4/strided/kernel*'
_output_shapes
:@ђ
н
decoder/layer_4/strided/kernelVarHandleOp*1
_class'
%#loc:@decoder/layer_4/strided/kernel*
dtype0*
_output_shapes
: *
shape:@ђ*/
shared_name decoder/layer_4/strided/kernel
Ї
?decoder/layer_4/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_4/strided/kernel*
_output_shapes
: 
А
%decoder/layer_4/strided/kernel/AssignAssignVariableOpdecoder/layer_4/strided/kernel9decoder/layer_4/strided/kernel/Initializer/random_uniform*
dtype0
џ
2decoder/layer_4/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_4/strided/kernel*
dtype0*'
_output_shapes
:@ђ
г
.decoder/layer_4/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_4/strided/bias*
valueB@*    *
dtype0*
_output_shapes
:@
┴
decoder/layer_4/strided/biasVarHandleOp*-
shared_namedecoder/layer_4/strided/bias*/
_class%
#!loc:@decoder/layer_4/strided/bias*
dtype0*
_output_shapes
: *
shape:@
Ѕ
=decoder/layer_4/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_4/strided/bias*
_output_shapes
: 
њ
#decoder/layer_4/strided/bias/AssignAssignVariableOpdecoder/layer_4/strided/bias.decoder/layer_4/strided/bias/Initializer/zeros*
dtype0
Ѕ
0decoder/layer_4/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_4/strided/bias*
dtype0*
_output_shapes
:@
e
decoder/layer_4/strided/ShapeShapedecoder/layer_3/ln/add_1*
_output_shapes
:*
T0
u
+decoder/layer_4/strided/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-decoder/layer_4/strided/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-decoder/layer_4/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_4/strided/strided_sliceStridedSlicedecoder/layer_4/strided/Shape+decoder/layer_4/strided/strided_slice/stack-decoder/layer_4/strided/strided_slice/stack_1-decoder/layer_4/strided/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
w
-decoder/layer_4/strided/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_4/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_4/strided/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_4/strided/strided_slice_1StridedSlicedecoder/layer_4/strided/Shape-decoder/layer_4/strided/strided_slice_1/stack/decoder/layer_4/strided/strided_slice_1/stack_1/decoder/layer_4/strided/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_4/strided/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_4/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_4/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_4/strided/strided_slice_2StridedSlicedecoder/layer_4/strided/Shape-decoder/layer_4/strided/strided_slice_2/stack/decoder/layer_4/strided/strided_slice_2/stack_1/decoder/layer_4/strided/strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
_
decoder/layer_4/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_4/strided/mulMul'decoder/layer_4/strided/strided_slice_1decoder/layer_4/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_4/strided/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Ј
decoder/layer_4/strided/mul_1Mul'decoder/layer_4/strided/strided_slice_2decoder/layer_4/strided/mul_1/y*
T0*
_output_shapes
: 
a
decoder/layer_4/strided/stack/3Const*
value	B :@*
dtype0*
_output_shapes
: 
О
decoder/layer_4/strided/stackPack%decoder/layer_4/strided/strided_slicedecoder/layer_4/strided/muldecoder/layer_4/strided/mul_1decoder/layer_4/strided/stack/3*
N*
_output_shapes
:*
T0
w
-decoder/layer_4/strided/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/decoder/layer_4/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_4/strided/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Г
'decoder/layer_4/strided/strided_slice_3StridedSlicedecoder/layer_4/strided/stack-decoder/layer_4/strided/strided_slice_3/stack/decoder/layer_4/strided/strided_slice_3/stack_1/decoder/layer_4/strided/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Ъ
7decoder/layer_4/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_4/strided/kernel*
dtype0*'
_output_shapes
:@ђ
Њ
(decoder/layer_4/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_4/strided/stack7decoder/layer_4/strided/conv2d_transpose/ReadVariableOpdecoder/layer_3/ln/add_1*
paddingSAME*
T0*
strides
*/
_output_shapes
:           @
Є
.decoder/layer_4/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_4/strided/bias*
dtype0*
_output_shapes
:@
Й
decoder/layer_4/strided/BiasAddBiasAdd(decoder/layer_4/strided/conv2d_transpose.decoder/layer_4/strided/BiasAdd/ReadVariableOp*/
_output_shapes
:           @*
T0
n
decoder/layer_4/strided/Shape_1Shapedecoder/layer_4/strided/BiasAdd*
T0*
_output_shapes
:
w
-decoder/layer_4/strided/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/decoder/layer_4/strided/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_4/strided/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
»
'decoder/layer_4/strided/strided_slice_4StridedSlicedecoder/layer_4/strided/Shape_1-decoder/layer_4/strided/strided_slice_4/stack/decoder/layer_4/strided/strided_slice_4/stack_1/decoder/layer_4/strided/strided_slice_4/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
i
'decoder/layer_4/strided/Reshape/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
i
'decoder/layer_4/strided/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B : 
r
'decoder/layer_4/strided/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
i
'decoder/layer_4/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_4/strided/Reshape/shapePack'decoder/layer_4/strided/strided_slice_4'decoder/layer_4/strided/Reshape/shape/1'decoder/layer_4/strided/Reshape/shape/2'decoder/layer_4/strided/Reshape/shape/3'decoder/layer_4/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_4/strided/ReshapeReshapedecoder/layer_4/strided/BiasAdd%decoder/layer_4/strided/Reshape/shape*<
_output_shapes*
(:&                    *
T0
_
decoder/layer_4/strided/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_4/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_4/strided/splitSplit'decoder/layer_4/strided/split/split_dimdecoder/layer_4/strided/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                    :&                    
ѕ
decoder/layer_4/strided/EluEludecoder/layer_4/strided/split*<
_output_shapes*
(:&                    *
T0
і
decoder/layer_4/strided/NegNegdecoder/layer_4/strided/split:1*
T0*<
_output_shapes*
(:&                    
ѕ
decoder/layer_4/strided/Elu_1Eludecoder/layer_4/strided/Neg*
T0*<
_output_shapes*
(:&                    
і
decoder/layer_4/strided/Neg_1Negdecoder/layer_4/strided/Elu_1*<
_output_shapes*
(:&                    *
T0
n
#decoder/layer_4/strided/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
decoder/layer_4/strided/concatConcatV2decoder/layer_4/strided/Eludecoder/layer_4/strided/Neg_1#decoder/layer_4/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&                    
k
)decoder/layer_4/strided/Reshape_1/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
k
)decoder/layer_4/strided/Reshape_1/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
k
)decoder/layer_4/strided/Reshape_1/shape/3Const*
value	B :@*
dtype0*
_output_shapes
: 
Є
'decoder/layer_4/strided/Reshape_1/shapePack'decoder/layer_4/strided/strided_slice_4)decoder/layer_4/strided/Reshape_1/shape/1)decoder/layer_4/strided/Reshape_1/shape/2)decoder/layer_4/strided/Reshape_1/shape/3*
N*
_output_shapes
:*
T0
»
!decoder/layer_4/strided/Reshape_1Reshapedecoder/layer_4/strided/concat'decoder/layer_4/strided/Reshape_1/shape*
T0*/
_output_shapes
:           @
т
Ldecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*
valueB
 *Хh¤й*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*
valueB
 *Хh¤=*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:@
╩
Jdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel
С
Jdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*&
_output_shapes
:@
о
Fdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*&
_output_shapes
:@
Щ
+decoder/layer_4/residual_0/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_4/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:@*<
shared_name-+decoder/layer_4/residual_0/depthwise_kernel
Д
Ldecoder/layer_4/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_4/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_4/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_4/residual_0/depthwise_kernelFdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_4/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_4/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:@
т
Ldecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*%
valueB"      @   ђ   *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*
valueB
 *з5Й
¤
Jdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*
valueB
 *з5>
▓
Tdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*
dtype0*'
_output_shapes
:@ђ
╩
Jdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/sub*'
_output_shapes
:@ђ*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel
О
Fdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*'
_output_shapes
:@ђ
ч
+decoder/layer_4/residual_0/pointwise_kernelVarHandleOp*
shape:@ђ*<
shared_name-+decoder/layer_4/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_4/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: 
Д
Ldecoder/layer_4/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_4/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_4/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_4/residual_0/pointwise_kernelFdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_4/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_4/residual_0/pointwise_kernel*
dtype0*'
_output_shapes
:@ђ
┤
1decoder/layer_4/residual_0/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_4/residual_0/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
╦
decoder/layer_4/residual_0/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*0
shared_name!decoder/layer_4/residual_0/bias*2
_class(
&$loc:@decoder/layer_4/residual_0/bias
Ј
@decoder/layer_4/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_4/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_4/residual_0/bias/AssignAssignVariableOpdecoder/layer_4/residual_0/bias1decoder/layer_4/residual_0/bias/Initializer/zeros*
dtype0
љ
3decoder/layer_4/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_4/residual_0/bias*
dtype0*
_output_shapes	
:ђ
«
:decoder/layer_4/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_4/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:@
▒
<decoder/layer_4/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_4/residual_0/pointwise_kernel*
dtype0*'
_output_shapes
:@ђ
і
1decoder/layer_4/residual_0/separable_conv2d/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
і
9decoder/layer_4/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ј
5decoder/layer_4/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_4/strided/Reshape_1:decoder/layer_4/residual_0/separable_conv2d/ReadVariableOp*
T0*
strides
*/
_output_shapes
:           @*
paddingSAME
ј
+decoder/layer_4/residual_0/separable_conv2dConv2D5decoder/layer_4/residual_0/separable_conv2d/depthwise<decoder/layer_4/residual_0/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*0
_output_shapes
:           ђ
ј
1decoder/layer_4/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_4/residual_0/bias*
dtype0*
_output_shapes	
:ђ
╚
"decoder/layer_4/residual_0/BiasAddBiasAdd+decoder/layer_4/residual_0/separable_conv2d1decoder/layer_4/residual_0/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:           ђ
r
 decoder/layer_4/residual_0/ShapeShape"decoder/layer_4/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_4/residual_0/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
z
0decoder/layer_4/residual_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_4/residual_0/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
┤
(decoder/layer_4/residual_0/strided_sliceStridedSlice decoder/layer_4/residual_0/Shape.decoder/layer_4/residual_0/strided_slice/stack0decoder/layer_4/residual_0/strided_slice/stack_10decoder/layer_4/residual_0/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
l
*decoder/layer_4/residual_0/Reshape/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
l
*decoder/layer_4/residual_0/Reshape/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
u
*decoder/layer_4/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_4/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_4/residual_0/Reshape/shapePack(decoder/layer_4/residual_0/strided_slice*decoder/layer_4/residual_0/Reshape/shape/1*decoder/layer_4/residual_0/Reshape/shape/2*decoder/layer_4/residual_0/Reshape/shape/3*decoder/layer_4/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_4/residual_0/ReshapeReshape"decoder/layer_4/residual_0/BiasAdd(decoder/layer_4/residual_0/Reshape/shape*
T0*<
_output_shapes*
(:&                    
b
 decoder/layer_4/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_4/residual_0/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_4/residual_0/splitSplit*decoder/layer_4/residual_0/split/split_dim"decoder/layer_4/residual_0/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                    :&                    
ј
decoder/layer_4/residual_0/EluElu decoder/layer_4/residual_0/split*<
_output_shapes*
(:&                    *
T0
љ
decoder/layer_4/residual_0/NegNeg"decoder/layer_4/residual_0/split:1*
T0*<
_output_shapes*
(:&                    
ј
 decoder/layer_4/residual_0/Elu_1Eludecoder/layer_4/residual_0/Neg*
T0*<
_output_shapes*
(:&                    
љ
 decoder/layer_4/residual_0/Neg_1Neg decoder/layer_4/residual_0/Elu_1*
T0*<
_output_shapes*
(:&                    
q
&decoder/layer_4/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_4/residual_0/concatConcatV2decoder/layer_4/residual_0/Elu decoder/layer_4/residual_0/Neg_1&decoder/layer_4/residual_0/concat/axis*
T0*
N*<
_output_shapes*
(:&                    
n
,decoder/layer_4/residual_0/Reshape_1/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
n
,decoder/layer_4/residual_0/Reshape_1/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
o
,decoder/layer_4/residual_0/Reshape_1/shape/3Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_4/residual_0/Reshape_1/shapePack(decoder/layer_4/residual_0/strided_slice,decoder/layer_4/residual_0/Reshape_1/shape/1,decoder/layer_4/residual_0/Reshape_1/shape/2,decoder/layer_4/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
╣
$decoder/layer_4/residual_0/Reshape_1Reshape!decoder/layer_4/residual_0/concat*decoder/layer_4/residual_0/Reshape_1/shape*
T0*0
_output_shapes
:           ђ
т
Ldecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*%
valueB"      ђ      *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*
valueB
 *I:Њй*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*
valueB
 *I:Њ=
▓
Tdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:ђ*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel
╩
Jdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*
_output_shapes
: 
т
Jdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*'
_output_shapes
:ђ
О
Fdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*'
_output_shapes
:ђ
ч
+decoder/layer_4/residual_1/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_4/residual_1/depthwise_kernel*>
_class4
20loc:@decoder/layer_4/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
Д
Ldecoder/layer_4/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_4/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_4/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_4/residual_1/depthwise_kernelFdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_4/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_4/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
т
Ldecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel*%
valueB"      ђ   @   
¤
Jdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel*
valueB
 *з5Й*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel*
valueB
 *з5>*
dtype0*
_output_shapes
: 
▓
Tdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel*
dtype0*'
_output_shapes
:ђ@
╩
Jdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel
т
Jdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/sub*'
_output_shapes
:ђ@*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel
О
Fdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel*'
_output_shapes
:ђ@
ч
+decoder/layer_4/residual_1/pointwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ@*<
shared_name-+decoder/layer_4/residual_1/pointwise_kernel*>
_class4
20loc:@decoder/layer_4/residual_1/pointwise_kernel
Д
Ldecoder/layer_4/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_4/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_4/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_4/residual_1/pointwise_kernelFdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
┤
?decoder/layer_4/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_4/residual_1/pointwise_kernel*
dtype0*'
_output_shapes
:ђ@
▓
1decoder/layer_4/residual_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*2
_class(
&$loc:@decoder/layer_4/residual_1/bias*
valueB@*    
╩
decoder/layer_4/residual_1/biasVarHandleOp*
shape:@*0
shared_name!decoder/layer_4/residual_1/bias*2
_class(
&$loc:@decoder/layer_4/residual_1/bias*
dtype0*
_output_shapes
: 
Ј
@decoder/layer_4/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_4/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_4/residual_1/bias/AssignAssignVariableOpdecoder/layer_4/residual_1/bias1decoder/layer_4/residual_1/bias/Initializer/zeros*
dtype0
Ј
3decoder/layer_4/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_4/residual_1/bias*
dtype0*
_output_shapes
:@
»
:decoder/layer_4/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_4/residual_1/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
▒
<decoder/layer_4/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_4/residual_1/pointwise_kernel*
dtype0*'
_output_shapes
:ђ@
і
1decoder/layer_4/residual_1/separable_conv2d/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      ђ      
і
9decoder/layer_4/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Њ
5decoder/layer_4/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_4/residual_0/Reshape_1:decoder/layer_4/residual_1/separable_conv2d/ReadVariableOp*
T0*
strides
*0
_output_shapes
:           ђ*
paddingSAME
Ї
+decoder/layer_4/residual_1/separable_conv2dConv2D5decoder/layer_4/residual_1/separable_conv2d/depthwise<decoder/layer_4/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*/
_output_shapes
:           @
Ї
1decoder/layer_4/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_4/residual_1/bias*
dtype0*
_output_shapes
:@
К
"decoder/layer_4/residual_1/BiasAddBiasAdd+decoder/layer_4/residual_1/separable_conv2d1decoder/layer_4/residual_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:           @
r
 decoder/layer_4/residual_1/ShapeShape"decoder/layer_4/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_4/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_4/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_4/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_4/residual_1/strided_sliceStridedSlice decoder/layer_4/residual_1/Shape.decoder/layer_4/residual_1/strided_slice/stack0decoder/layer_4/residual_1/strided_slice/stack_10decoder/layer_4/residual_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
l
*decoder/layer_4/residual_1/Reshape/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
l
*decoder/layer_4/residual_1/Reshape/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
u
*decoder/layer_4/residual_1/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_4/residual_1/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_4/residual_1/Reshape/shapePack(decoder/layer_4/residual_1/strided_slice*decoder/layer_4/residual_1/Reshape/shape/1*decoder/layer_4/residual_1/Reshape/shape/2*decoder/layer_4/residual_1/Reshape/shape/3*decoder/layer_4/residual_1/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_4/residual_1/ReshapeReshape"decoder/layer_4/residual_1/BiasAdd(decoder/layer_4/residual_1/Reshape/shape*<
_output_shapes*
(:&                    *
T0
b
 decoder/layer_4/residual_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_4/residual_1/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_4/residual_1/splitSplit*decoder/layer_4/residual_1/split/split_dim"decoder/layer_4/residual_1/Reshape*
T0*
	num_split*d
_output_shapesR
P:&                    :&                    
ј
decoder/layer_4/residual_1/EluElu decoder/layer_4/residual_1/split*
T0*<
_output_shapes*
(:&                    
љ
decoder/layer_4/residual_1/NegNeg"decoder/layer_4/residual_1/split:1*
T0*<
_output_shapes*
(:&                    
ј
 decoder/layer_4/residual_1/Elu_1Eludecoder/layer_4/residual_1/Neg*
T0*<
_output_shapes*
(:&                    
љ
 decoder/layer_4/residual_1/Neg_1Neg decoder/layer_4/residual_1/Elu_1*<
_output_shapes*
(:&                    *
T0
q
&decoder/layer_4/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_4/residual_1/concatConcatV2decoder/layer_4/residual_1/Elu decoder/layer_4/residual_1/Neg_1&decoder/layer_4/residual_1/concat/axis*
T0*
N*<
_output_shapes*
(:&                    
n
,decoder/layer_4/residual_1/Reshape_1/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
n
,decoder/layer_4/residual_1/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B : 
n
,decoder/layer_4/residual_1/Reshape_1/shape/3Const*
value	B :@*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_4/residual_1/Reshape_1/shapePack(decoder/layer_4/residual_1/strided_slice,decoder/layer_4/residual_1/Reshape_1/shape/1,decoder/layer_4/residual_1/Reshape_1/shape/2,decoder/layer_4/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
И
$decoder/layer_4/residual_1/Reshape_1Reshape!decoder/layer_4/residual_1/concat*decoder/layer_4/residual_1/Reshape_1/shape*/
_output_shapes
:           @*
T0
Ъ
decoder/layer_4/addAddV2!decoder/layer_4/strided/Reshape_1$decoder/layer_4/residual_1/Reshape_1*
T0*/
_output_shapes
:           @
X
decoder/layer_4/ShapeShapedecoder/layer_4/add*
_output_shapes
:*
T0
m
#decoder/layer_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
decoder/layer_4/strided_sliceStridedSlicedecoder/layer_4/Shape#decoder/layer_4/strided_slice/stack%decoder/layer_4/strided_slice/stack_1%decoder/layer_4/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
╣
4decoder/layer_4/ln/layer_norm_scale/Initializer/onesConst*6
_class,
*(loc:@decoder/layer_4/ln/layer_norm_scale*
valueB@*  ђ?*
dtype0*
_output_shapes
:@
о
#decoder/layer_4/ln/layer_norm_scaleVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*4
shared_name%#decoder/layer_4/ln/layer_norm_scale*6
_class,
*(loc:@decoder/layer_4/ln/layer_norm_scale
Ќ
Ddecoder/layer_4/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_4/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_4/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_4/ln/layer_norm_scale4decoder/layer_4/ln/layer_norm_scale/Initializer/ones*
dtype0
Ќ
7decoder/layer_4/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_4/ln/layer_norm_scale*
dtype0*
_output_shapes
:@
И
4decoder/layer_4/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_4/ln/layer_norm_bias*
valueB@*    *
dtype0*
_output_shapes
:@
М
"decoder/layer_4/ln/layer_norm_biasVarHandleOp*3
shared_name$"decoder/layer_4/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_4/ln/layer_norm_bias*
dtype0*
_output_shapes
: *
shape:@
Ћ
Cdecoder/layer_4/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_4/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_4/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_4/ln/layer_norm_bias4decoder/layer_4/ln/layer_norm_bias/Initializer/zeros*
dtype0
Ћ
6decoder/layer_4/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_4/ln/layer_norm_bias*
dtype0*
_output_shapes
:@
]
decoder/layer_4/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ђ
!decoder/layer_4/ln/ReadVariableOpReadVariableOp#decoder/layer_4/ln/layer_norm_scale*
dtype0*
_output_shapes
:@
ѓ
#decoder/layer_4/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_4/ln/layer_norm_bias*
dtype0*
_output_shapes
:@
|
)decoder/layer_4/ln/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
         
ф
decoder/layer_4/ln/MeanMeandecoder/layer_4/add)decoder/layer_4/ln/Mean/reduction_indices*
T0*/
_output_shapes
:           *
	keep_dims(
А
$decoder/layer_4/ln/SquaredDifferenceSquaredDifferencedecoder/layer_4/adddecoder/layer_4/ln/Mean*
T0*/
_output_shapes
:           @
~
+decoder/layer_4/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┐
decoder/layer_4/ln/Mean_1Mean$decoder/layer_4/ln/SquaredDifference+decoder/layer_4/ln/Mean_1/reduction_indices*
T0*/
_output_shapes
:           *
	keep_dims(
Ё
decoder/layer_4/ln/subSubdecoder/layer_4/adddecoder/layer_4/ln/Mean*
T0*/
_output_shapes
:           @
ј
decoder/layer_4/ln/addAddV2decoder/layer_4/ln/Mean_1decoder/layer_4/ln/Const*
T0*/
_output_shapes
:           
s
decoder/layer_4/ln/RsqrtRsqrtdecoder/layer_4/ln/add*
T0*/
_output_shapes
:           
Ѕ
decoder/layer_4/ln/mulMuldecoder/layer_4/ln/subdecoder/layer_4/ln/Rsqrt*/
_output_shapes
:           @*
T0
ћ
decoder/layer_4/ln/mul_1Muldecoder/layer_4/ln/mul!decoder/layer_4/ln/ReadVariableOp*/
_output_shapes
:           @*
T0
џ
decoder/layer_4/ln/add_1AddV2decoder/layer_4/ln/mul_1#decoder/layer_4/ln/ReadVariableOp_1*/
_output_shapes
:           @*
T0
╦
?decoder/layer_5/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_5/strided/kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
х
=decoder/layer_5/strided/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@decoder/layer_5/strided/kernel*
valueB
 *  ђй
х
=decoder/layer_5/strided/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@decoder/layer_5/strided/kernel*
valueB
 *  ђ=*
dtype0*
_output_shapes
: 
і
Gdecoder/layer_5/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_5/strided/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*
T0*1
_class'
%#loc:@decoder/layer_5/strided/kernel
ќ
=decoder/layer_5/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_5/strided/kernel/Initializer/random_uniform/max=decoder/layer_5/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_5/strided/kernel*
_output_shapes
: 
░
=decoder/layer_5/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_5/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_5/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_5/strided/kernel*&
_output_shapes
: @
б
9decoder/layer_5/strided/kernel/Initializer/random_uniformAdd=decoder/layer_5/strided/kernel/Initializer/random_uniform/mul=decoder/layer_5/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_5/strided/kernel*&
_output_shapes
: @
М
decoder/layer_5/strided/kernelVarHandleOp*
shape: @*/
shared_name decoder/layer_5/strided/kernel*1
_class'
%#loc:@decoder/layer_5/strided/kernel*
dtype0*
_output_shapes
: 
Ї
?decoder/layer_5/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_5/strided/kernel*
_output_shapes
: 
А
%decoder/layer_5/strided/kernel/AssignAssignVariableOpdecoder/layer_5/strided/kernel9decoder/layer_5/strided/kernel/Initializer/random_uniform*
dtype0
Ў
2decoder/layer_5/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_5/strided/kernel*
dtype0*&
_output_shapes
: @
г
.decoder/layer_5/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_5/strided/bias*
valueB *    *
dtype0*
_output_shapes
: 
┴
decoder/layer_5/strided/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *-
shared_namedecoder/layer_5/strided/bias*/
_class%
#!loc:@decoder/layer_5/strided/bias
Ѕ
=decoder/layer_5/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_5/strided/bias*
_output_shapes
: 
њ
#decoder/layer_5/strided/bias/AssignAssignVariableOpdecoder/layer_5/strided/bias.decoder/layer_5/strided/bias/Initializer/zeros*
dtype0
Ѕ
0decoder/layer_5/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_5/strided/bias*
dtype0*
_output_shapes
: 
e
decoder/layer_5/strided/ShapeShapedecoder/layer_4/ln/add_1*
T0*
_output_shapes
:
u
+decoder/layer_5/strided/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-decoder/layer_5/strided/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-decoder/layer_5/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_5/strided/strided_sliceStridedSlicedecoder/layer_5/strided/Shape+decoder/layer_5/strided/strided_slice/stack-decoder/layer_5/strided/strided_slice/stack_1-decoder/layer_5/strided/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
w
-decoder/layer_5/strided/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_5/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_5/strided/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_5/strided/strided_slice_1StridedSlicedecoder/layer_5/strided/Shape-decoder/layer_5/strided/strided_slice_1/stack/decoder/layer_5/strided/strided_slice_1/stack_1/decoder/layer_5/strided/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
w
-decoder/layer_5/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_5/strided/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_5/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_5/strided/strided_slice_2StridedSlicedecoder/layer_5/strided/Shape-decoder/layer_5/strided/strided_slice_2/stack/decoder/layer_5/strided/strided_slice_2/stack_1/decoder/layer_5/strided/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
_
decoder/layer_5/strided/mul/yConst*
dtype0*
_output_shapes
: *
value	B :
І
decoder/layer_5/strided/mulMul'decoder/layer_5/strided/strided_slice_1decoder/layer_5/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_5/strided/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Ј
decoder/layer_5/strided/mul_1Mul'decoder/layer_5/strided/strided_slice_2decoder/layer_5/strided/mul_1/y*
T0*
_output_shapes
: 
a
decoder/layer_5/strided/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
О
decoder/layer_5/strided/stackPack%decoder/layer_5/strided/strided_slicedecoder/layer_5/strided/muldecoder/layer_5/strided/mul_1decoder/layer_5/strided/stack/3*
N*
_output_shapes
:*
T0
w
-decoder/layer_5/strided/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_5/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_5/strided/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_5/strided/strided_slice_3StridedSlicedecoder/layer_5/strided/stack-decoder/layer_5/strided/strided_slice_3/stack/decoder/layer_5/strided/strided_slice_3/stack_1/decoder/layer_5/strided/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
ъ
7decoder/layer_5/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_5/strided/kernel*
dtype0*&
_output_shapes
: @
Њ
(decoder/layer_5/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_5/strided/stack7decoder/layer_5/strided/conv2d_transpose/ReadVariableOpdecoder/layer_4/ln/add_1*
T0*
strides
*/
_output_shapes
:         @@ *
paddingSAME
Є
.decoder/layer_5/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_5/strided/bias*
dtype0*
_output_shapes
: 
Й
decoder/layer_5/strided/BiasAddBiasAdd(decoder/layer_5/strided/conv2d_transpose.decoder/layer_5/strided/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:         @@ 
n
decoder/layer_5/strided/Shape_1Shapedecoder/layer_5/strided/BiasAdd*
_output_shapes
:*
T0
w
-decoder/layer_5/strided/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_5/strided/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/decoder/layer_5/strided/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
»
'decoder/layer_5/strided/strided_slice_4StridedSlicedecoder/layer_5/strided/Shape_1-decoder/layer_5/strided/strided_slice_4/stack/decoder/layer_5/strided/strided_slice_4/stack_1/decoder/layer_5/strided/strided_slice_4/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
i
'decoder/layer_5/strided/Reshape/shape/1Const*
value	B :@*
dtype0*
_output_shapes
: 
i
'decoder/layer_5/strided/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :@
r
'decoder/layer_5/strided/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
i
'decoder/layer_5/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_5/strided/Reshape/shapePack'decoder/layer_5/strided/strided_slice_4'decoder/layer_5/strided/Reshape/shape/1'decoder/layer_5/strided/Reshape/shape/2'decoder/layer_5/strided/Reshape/shape/3'decoder/layer_5/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╣
decoder/layer_5/strided/ReshapeReshapedecoder/layer_5/strided/BiasAdd%decoder/layer_5/strided/Reshape/shape*<
_output_shapes*
(:&         @@         *
T0
_
decoder/layer_5/strided/ConstConst*
dtype0*
_output_shapes
: *
value	B :
r
'decoder/layer_5/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
­
decoder/layer_5/strided/splitSplit'decoder/layer_5/strided/split/split_dimdecoder/layer_5/strided/Reshape*
T0*
	num_split*d
_output_shapesR
P:&         @@         :&         @@         
ѕ
decoder/layer_5/strided/EluEludecoder/layer_5/strided/split*
T0*<
_output_shapes*
(:&         @@         
і
decoder/layer_5/strided/NegNegdecoder/layer_5/strided/split:1*
T0*<
_output_shapes*
(:&         @@         
ѕ
decoder/layer_5/strided/Elu_1Eludecoder/layer_5/strided/Neg*
T0*<
_output_shapes*
(:&         @@         
і
decoder/layer_5/strided/Neg_1Negdecoder/layer_5/strided/Elu_1*<
_output_shapes*
(:&         @@         *
T0
n
#decoder/layer_5/strided/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
decoder/layer_5/strided/concatConcatV2decoder/layer_5/strided/Eludecoder/layer_5/strided/Neg_1#decoder/layer_5/strided/concat/axis*
T0*
N*<
_output_shapes*
(:&         @@         
k
)decoder/layer_5/strided/Reshape_1/shape/1Const*
value	B :@*
dtype0*
_output_shapes
: 
k
)decoder/layer_5/strided/Reshape_1/shape/2Const*
value	B :@*
dtype0*
_output_shapes
: 
k
)decoder/layer_5/strided/Reshape_1/shape/3Const*
value	B : *
dtype0*
_output_shapes
: 
Є
'decoder/layer_5/strided/Reshape_1/shapePack'decoder/layer_5/strided/strided_slice_4)decoder/layer_5/strided/Reshape_1/shape/1)decoder/layer_5/strided/Reshape_1/shape/2)decoder/layer_5/strided/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
»
!decoder/layer_5/strided/Reshape_1Reshapedecoder/layer_5/strided/concat'decoder/layer_5/strided/Reshape_1/shape*
T0*/
_output_shapes
:         @@ 
т
Ldecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*%
valueB"             *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*
valueB
 *ЄІЙ*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*
valueB
 *ЄІ>
▒
Tdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
: 
╩
Jdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel
о
Fdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*&
_output_shapes
: 
Щ
+decoder/layer_5/residual_0/depthwise_kernelVarHandleOp*<
shared_name-+decoder/layer_5/residual_0/depthwise_kernel*>
_class4
20loc:@decoder/layer_5/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: *
shape: 
Д
Ldecoder/layer_5/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_5/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_5/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_5/residual_0/depthwise_kernelFdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_5/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_5/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
: 
т
Ldecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*
valueB
 *  ђЙ*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: @
╩
Jdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*&
_output_shapes
: @
о
Fdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*&
_output_shapes
: @
Щ
+decoder/layer_5/residual_0/pointwise_kernelVarHandleOp*<
shared_name-+decoder/layer_5/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_5/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: *
shape: @
Д
Ldecoder/layer_5/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_5/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_5/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_5/residual_0/pointwise_kernelFdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_5/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_5/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: @
▓
1decoder/layer_5/residual_0/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*2
_class(
&$loc:@decoder/layer_5/residual_0/bias*
valueB@*    
╩
decoder/layer_5/residual_0/biasVarHandleOp*
shape:@*0
shared_name!decoder/layer_5/residual_0/bias*2
_class(
&$loc:@decoder/layer_5/residual_0/bias*
dtype0*
_output_shapes
: 
Ј
@decoder/layer_5/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_5/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_5/residual_0/bias/AssignAssignVariableOpdecoder/layer_5/residual_0/bias1decoder/layer_5/residual_0/bias/Initializer/zeros*
dtype0
Ј
3decoder/layer_5/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_5/residual_0/bias*
dtype0*
_output_shapes
:@
«
:decoder/layer_5/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_5/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
: 
░
<decoder/layer_5/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_5/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: @
і
1decoder/layer_5/residual_0/separable_conv2d/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
і
9decoder/layer_5/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ј
5decoder/layer_5/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_5/strided/Reshape_1:decoder/layer_5/residual_0/separable_conv2d/ReadVariableOp*
strides
*/
_output_shapes
:         @@ *
paddingSAME*
T0
Ї
+decoder/layer_5/residual_0/separable_conv2dConv2D5decoder/layer_5/residual_0/separable_conv2d/depthwise<decoder/layer_5/residual_0/separable_conv2d/ReadVariableOp_1*
strides
*/
_output_shapes
:         @@@*
paddingVALID*
T0
Ї
1decoder/layer_5/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_5/residual_0/bias*
dtype0*
_output_shapes
:@
К
"decoder/layer_5/residual_0/BiasAddBiasAdd+decoder/layer_5/residual_0/separable_conv2d1decoder/layer_5/residual_0/BiasAdd/ReadVariableOp*/
_output_shapes
:         @@@*
T0
r
 decoder/layer_5/residual_0/ShapeShape"decoder/layer_5/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_5/residual_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_5/residual_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_5/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_5/residual_0/strided_sliceStridedSlice decoder/layer_5/residual_0/Shape.decoder/layer_5/residual_0/strided_slice/stack0decoder/layer_5/residual_0/strided_slice/stack_10decoder/layer_5/residual_0/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
l
*decoder/layer_5/residual_0/Reshape/shape/1Const*
value	B :@*
dtype0*
_output_shapes
: 
l
*decoder/layer_5/residual_0/Reshape/shape/2Const*
value	B :@*
dtype0*
_output_shapes
: 
u
*decoder/layer_5/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_5/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_5/residual_0/Reshape/shapePack(decoder/layer_5/residual_0/strided_slice*decoder/layer_5/residual_0/Reshape/shape/1*decoder/layer_5/residual_0/Reshape/shape/2*decoder/layer_5/residual_0/Reshape/shape/3*decoder/layer_5/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
┬
"decoder/layer_5/residual_0/ReshapeReshape"decoder/layer_5/residual_0/BiasAdd(decoder/layer_5/residual_0/Reshape/shape*
T0*<
_output_shapes*
(:&         @@         
b
 decoder/layer_5/residual_0/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_5/residual_0/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
 decoder/layer_5/residual_0/splitSplit*decoder/layer_5/residual_0/split/split_dim"decoder/layer_5/residual_0/Reshape*
T0*
	num_split*d
_output_shapesR
P:&         @@         :&         @@         
ј
decoder/layer_5/residual_0/EluElu decoder/layer_5/residual_0/split*
T0*<
_output_shapes*
(:&         @@         
љ
decoder/layer_5/residual_0/NegNeg"decoder/layer_5/residual_0/split:1*
T0*<
_output_shapes*
(:&         @@         
ј
 decoder/layer_5/residual_0/Elu_1Eludecoder/layer_5/residual_0/Neg*
T0*<
_output_shapes*
(:&         @@         
љ
 decoder/layer_5/residual_0/Neg_1Neg decoder/layer_5/residual_0/Elu_1*
T0*<
_output_shapes*
(:&         @@         
q
&decoder/layer_5/residual_0/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
у
!decoder/layer_5/residual_0/concatConcatV2decoder/layer_5/residual_0/Elu decoder/layer_5/residual_0/Neg_1&decoder/layer_5/residual_0/concat/axis*
T0*
N*<
_output_shapes*
(:&         @@         
n
,decoder/layer_5/residual_0/Reshape_1/shape/1Const*
value	B :@*
dtype0*
_output_shapes
: 
n
,decoder/layer_5/residual_0/Reshape_1/shape/2Const*
value	B :@*
dtype0*
_output_shapes
: 
n
,decoder/layer_5/residual_0/Reshape_1/shape/3Const*
value	B :@*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_5/residual_0/Reshape_1/shapePack(decoder/layer_5/residual_0/strided_slice,decoder/layer_5/residual_0/Reshape_1/shape/1,decoder/layer_5/residual_0/Reshape_1/shape/2,decoder/layer_5/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
И
$decoder/layer_5/residual_0/Reshape_1Reshape!decoder/layer_5/residual_0/concat*decoder/layer_5/residual_0/Reshape_1/shape*
T0*/
_output_shapes
:         @@@
т
Ldecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*
valueB
 *Хh¤й*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*
valueB
 *Хh¤=*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*
dtype0*&
_output_shapes
:@
╩
Jdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*&
_output_shapes
:@
о
Fdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform/min*&
_output_shapes
:@*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel
Щ
+decoder/layer_5/residual_1/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_5/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:@*<
shared_name-+decoder/layer_5/residual_1/depthwise_kernel
Д
Ldecoder/layer_5/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_5/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_5/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_5/residual_1/depthwise_kernelFdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_5/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_5/residual_1/depthwise_kernel*
dtype0*&
_output_shapes
:@
т
Ldecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*%
valueB"      @       *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*
valueB
 *  ђЙ*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
:@ 
╩
Jdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*&
_output_shapes
:@ 
о
Fdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*&
_output_shapes
:@ 
Щ
+decoder/layer_5/residual_1/pointwise_kernelVarHandleOp*<
shared_name-+decoder/layer_5/residual_1/pointwise_kernel*>
_class4
20loc:@decoder/layer_5/residual_1/pointwise_kernel*
dtype0*
_output_shapes
: *
shape:@ 
Д
Ldecoder/layer_5/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_5/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_5/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_5/residual_1/pointwise_kernelFdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_5/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_5/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
:@ 
▓
1decoder/layer_5/residual_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@decoder/layer_5/residual_1/bias*
valueB *    
╩
decoder/layer_5/residual_1/biasVarHandleOp*0
shared_name!decoder/layer_5/residual_1/bias*2
_class(
&$loc:@decoder/layer_5/residual_1/bias*
dtype0*
_output_shapes
: *
shape: 
Ј
@decoder/layer_5/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_5/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_5/residual_1/bias/AssignAssignVariableOpdecoder/layer_5/residual_1/bias1decoder/layer_5/residual_1/bias/Initializer/zeros*
dtype0
Ј
3decoder/layer_5/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_5/residual_1/bias*
dtype0*
_output_shapes
: 
«
:decoder/layer_5/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_5/residual_1/depthwise_kernel*
dtype0*&
_output_shapes
:@
░
<decoder/layer_5/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_5/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
:@ 
і
1decoder/layer_5/residual_1/separable_conv2d/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
і
9decoder/layer_5/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
њ
5decoder/layer_5/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_5/residual_0/Reshape_1:decoder/layer_5/residual_1/separable_conv2d/ReadVariableOp*
strides
*/
_output_shapes
:         @@@*
paddingSAME*
T0
Ї
+decoder/layer_5/residual_1/separable_conv2dConv2D5decoder/layer_5/residual_1/separable_conv2d/depthwise<decoder/layer_5/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*/
_output_shapes
:         @@ 
Ї
1decoder/layer_5/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_5/residual_1/bias*
dtype0*
_output_shapes
: 
К
"decoder/layer_5/residual_1/BiasAddBiasAdd+decoder/layer_5/residual_1/separable_conv2d1decoder/layer_5/residual_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:         @@ 
r
 decoder/layer_5/residual_1/ShapeShape"decoder/layer_5/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_5/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_5/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_5/residual_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
┤
(decoder/layer_5/residual_1/strided_sliceStridedSlice decoder/layer_5/residual_1/Shape.decoder/layer_5/residual_1/strided_slice/stack0decoder/layer_5/residual_1/strided_slice/stack_10decoder/layer_5/residual_1/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
l
*decoder/layer_5/residual_1/Reshape/shape/1Const*
value	B :@*
dtype0*
_output_shapes
: 
l
*decoder/layer_5/residual_1/Reshape/shape/2Const*
value	B :@*
dtype0*
_output_shapes
: 
u
*decoder/layer_5/residual_1/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
valueB :
         
l
*decoder/layer_5/residual_1/Reshape/shape/4Const*
dtype0*
_output_shapes
: *
value	B :
И
(decoder/layer_5/residual_1/Reshape/shapePack(decoder/layer_5/residual_1/strided_slice*decoder/layer_5/residual_1/Reshape/shape/1*decoder/layer_5/residual_1/Reshape/shape/2*decoder/layer_5/residual_1/Reshape/shape/3*decoder/layer_5/residual_1/Reshape/shape/4*
N*
_output_shapes
:*
T0
┬
"decoder/layer_5/residual_1/ReshapeReshape"decoder/layer_5/residual_1/BiasAdd(decoder/layer_5/residual_1/Reshape/shape*<
_output_shapes*
(:&         @@         *
T0
b
 decoder/layer_5/residual_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_5/residual_1/split/split_dimConst*
dtype0*
_output_shapes
: *
valueB :
         
щ
 decoder/layer_5/residual_1/splitSplit*decoder/layer_5/residual_1/split/split_dim"decoder/layer_5/residual_1/Reshape*
T0*
	num_split*d
_output_shapesR
P:&         @@         :&         @@         
ј
decoder/layer_5/residual_1/EluElu decoder/layer_5/residual_1/split*
T0*<
_output_shapes*
(:&         @@         
љ
decoder/layer_5/residual_1/NegNeg"decoder/layer_5/residual_1/split:1*
T0*<
_output_shapes*
(:&         @@         
ј
 decoder/layer_5/residual_1/Elu_1Eludecoder/layer_5/residual_1/Neg*<
_output_shapes*
(:&         @@         *
T0
љ
 decoder/layer_5/residual_1/Neg_1Neg decoder/layer_5/residual_1/Elu_1*<
_output_shapes*
(:&         @@         *
T0
q
&decoder/layer_5/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
!decoder/layer_5/residual_1/concatConcatV2decoder/layer_5/residual_1/Elu decoder/layer_5/residual_1/Neg_1&decoder/layer_5/residual_1/concat/axis*
N*<
_output_shapes*
(:&         @@         *
T0
n
,decoder/layer_5/residual_1/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value	B :@
n
,decoder/layer_5/residual_1/Reshape_1/shape/2Const*
value	B :@*
dtype0*
_output_shapes
: 
n
,decoder/layer_5/residual_1/Reshape_1/shape/3Const*
value	B : *
dtype0*
_output_shapes
: 
ћ
*decoder/layer_5/residual_1/Reshape_1/shapePack(decoder/layer_5/residual_1/strided_slice,decoder/layer_5/residual_1/Reshape_1/shape/1,decoder/layer_5/residual_1/Reshape_1/shape/2,decoder/layer_5/residual_1/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
И
$decoder/layer_5/residual_1/Reshape_1Reshape!decoder/layer_5/residual_1/concat*decoder/layer_5/residual_1/Reshape_1/shape*
T0*/
_output_shapes
:         @@ 
Ъ
decoder/layer_5/addAddV2!decoder/layer_5/strided/Reshape_1$decoder/layer_5/residual_1/Reshape_1*
T0*/
_output_shapes
:         @@ 
X
decoder/layer_5/ShapeShapedecoder/layer_5/add*
_output_shapes
:*
T0
m
#decoder/layer_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
decoder/layer_5/strided_sliceStridedSlicedecoder/layer_5/Shape#decoder/layer_5/strided_slice/stack%decoder/layer_5/strided_slice/stack_1%decoder/layer_5/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
╣
4decoder/layer_5/ln/layer_norm_scale/Initializer/onesConst*6
_class,
*(loc:@decoder/layer_5/ln/layer_norm_scale*
valueB *  ђ?*
dtype0*
_output_shapes
: 
о
#decoder/layer_5/ln/layer_norm_scaleVarHandleOp*6
_class,
*(loc:@decoder/layer_5/ln/layer_norm_scale*
dtype0*
_output_shapes
: *
shape: *4
shared_name%#decoder/layer_5/ln/layer_norm_scale
Ќ
Ddecoder/layer_5/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_5/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_5/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_5/ln/layer_norm_scale4decoder/layer_5/ln/layer_norm_scale/Initializer/ones*
dtype0
Ќ
7decoder/layer_5/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_5/ln/layer_norm_scale*
dtype0*
_output_shapes
: 
И
4decoder/layer_5/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_5/ln/layer_norm_bias*
valueB *    *
dtype0*
_output_shapes
: 
М
"decoder/layer_5/ln/layer_norm_biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *3
shared_name$"decoder/layer_5/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_5/ln/layer_norm_bias
Ћ
Cdecoder/layer_5/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_5/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_5/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_5/ln/layer_norm_bias4decoder/layer_5/ln/layer_norm_bias/Initializer/zeros*
dtype0
Ћ
6decoder/layer_5/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_5/ln/layer_norm_bias*
dtype0*
_output_shapes
: 
]
decoder/layer_5/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ђ
!decoder/layer_5/ln/ReadVariableOpReadVariableOp#decoder/layer_5/ln/layer_norm_scale*
dtype0*
_output_shapes
: 
ѓ
#decoder/layer_5/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_5/ln/layer_norm_bias*
dtype0*
_output_shapes
: 
|
)decoder/layer_5/ln/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
         
ф
decoder/layer_5/ln/MeanMeandecoder/layer_5/add)decoder/layer_5/ln/Mean/reduction_indices*
	keep_dims(*
T0*/
_output_shapes
:         @@
А
$decoder/layer_5/ln/SquaredDifferenceSquaredDifferencedecoder/layer_5/adddecoder/layer_5/ln/Mean*
T0*/
_output_shapes
:         @@ 
~
+decoder/layer_5/ln/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
         
┐
decoder/layer_5/ln/Mean_1Mean$decoder/layer_5/ln/SquaredDifference+decoder/layer_5/ln/Mean_1/reduction_indices*
T0*/
_output_shapes
:         @@*
	keep_dims(
Ё
decoder/layer_5/ln/subSubdecoder/layer_5/adddecoder/layer_5/ln/Mean*
T0*/
_output_shapes
:         @@ 
ј
decoder/layer_5/ln/addAddV2decoder/layer_5/ln/Mean_1decoder/layer_5/ln/Const*
T0*/
_output_shapes
:         @@
s
decoder/layer_5/ln/RsqrtRsqrtdecoder/layer_5/ln/add*
T0*/
_output_shapes
:         @@
Ѕ
decoder/layer_5/ln/mulMuldecoder/layer_5/ln/subdecoder/layer_5/ln/Rsqrt*
T0*/
_output_shapes
:         @@ 
ћ
decoder/layer_5/ln/mul_1Muldecoder/layer_5/ln/mul!decoder/layer_5/ln/ReadVariableOp*/
_output_shapes
:         @@ *
T0
џ
decoder/layer_5/ln/add_1AddV2decoder/layer_5/ln/mul_1#decoder/layer_5/ln/ReadVariableOp_1*
T0*/
_output_shapes
:         @@ 
╦
?decoder/layer_6/strided/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@decoder/layer_6/strided/kernel*%
valueB"             *
dtype0*
_output_shapes
:
х
=decoder/layer_6/strided/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@decoder/layer_6/strided/kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
х
=decoder/layer_6/strided/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@decoder/layer_6/strided/kernel*
valueB
 *зх=*
dtype0*
_output_shapes
: 
і
Gdecoder/layer_6/strided/kernel/Initializer/random_uniform/RandomUniformRandomUniform?decoder/layer_6/strided/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*1
_class'
%#loc:@decoder/layer_6/strided/kernel
ќ
=decoder/layer_6/strided/kernel/Initializer/random_uniform/subSub=decoder/layer_6/strided/kernel/Initializer/random_uniform/max=decoder/layer_6/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_6/strided/kernel*
_output_shapes
: 
░
=decoder/layer_6/strided/kernel/Initializer/random_uniform/mulMulGdecoder/layer_6/strided/kernel/Initializer/random_uniform/RandomUniform=decoder/layer_6/strided/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@decoder/layer_6/strided/kernel*&
_output_shapes
: 
б
9decoder/layer_6/strided/kernel/Initializer/random_uniformAdd=decoder/layer_6/strided/kernel/Initializer/random_uniform/mul=decoder/layer_6/strided/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@decoder/layer_6/strided/kernel*&
_output_shapes
: 
М
decoder/layer_6/strided/kernelVarHandleOp*1
_class'
%#loc:@decoder/layer_6/strided/kernel*
dtype0*
_output_shapes
: *
shape: */
shared_name decoder/layer_6/strided/kernel
Ї
?decoder/layer_6/strided/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_6/strided/kernel*
_output_shapes
: 
А
%decoder/layer_6/strided/kernel/AssignAssignVariableOpdecoder/layer_6/strided/kernel9decoder/layer_6/strided/kernel/Initializer/random_uniform*
dtype0
Ў
2decoder/layer_6/strided/kernel/Read/ReadVariableOpReadVariableOpdecoder/layer_6/strided/kernel*
dtype0*&
_output_shapes
: 
г
.decoder/layer_6/strided/bias/Initializer/zerosConst*/
_class%
#!loc:@decoder/layer_6/strided/bias*
valueB*    *
dtype0*
_output_shapes
:
┴
decoder/layer_6/strided/biasVarHandleOp*-
shared_namedecoder/layer_6/strided/bias*/
_class%
#!loc:@decoder/layer_6/strided/bias*
dtype0*
_output_shapes
: *
shape:
Ѕ
=decoder/layer_6/strided/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_6/strided/bias*
_output_shapes
: 
њ
#decoder/layer_6/strided/bias/AssignAssignVariableOpdecoder/layer_6/strided/bias.decoder/layer_6/strided/bias/Initializer/zeros*
dtype0
Ѕ
0decoder/layer_6/strided/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_6/strided/bias*
dtype0*
_output_shapes
:
e
decoder/layer_6/strided/ShapeShapedecoder/layer_5/ln/add_1*
T0*
_output_shapes
:
u
+decoder/layer_6/strided/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-decoder/layer_6/strided/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-decoder/layer_6/strided/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
%decoder/layer_6/strided/strided_sliceStridedSlicedecoder/layer_6/strided/Shape+decoder/layer_6/strided/strided_slice/stack-decoder/layer_6/strided/strided_slice/stack_1-decoder/layer_6/strided/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
-decoder/layer_6/strided/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Г
'decoder/layer_6/strided/strided_slice_1StridedSlicedecoder/layer_6/strided/Shape-decoder/layer_6/strided/strided_slice_1/stack/decoder/layer_6/strided/strided_slice_1/stack_1/decoder/layer_6/strided/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
w
-decoder/layer_6/strided/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_6/strided/strided_slice_2StridedSlicedecoder/layer_6/strided/Shape-decoder/layer_6/strided/strided_slice_2/stack/decoder/layer_6/strided/strided_slice_2/stack_1/decoder/layer_6/strided/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
_
decoder/layer_6/strided/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
decoder/layer_6/strided/mulMul'decoder/layer_6/strided/strided_slice_1decoder/layer_6/strided/mul/y*
T0*
_output_shapes
: 
a
decoder/layer_6/strided/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
decoder/layer_6/strided/mul_1Mul'decoder/layer_6/strided/strided_slice_2decoder/layer_6/strided/mul_1/y*
T0*
_output_shapes
: 
a
decoder/layer_6/strided/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
О
decoder/layer_6/strided/stackPack%decoder/layer_6/strided/strided_slicedecoder/layer_6/strided/muldecoder/layer_6/strided/mul_1decoder/layer_6/strided/stack/3*
T0*
N*
_output_shapes
:
w
-decoder/layer_6/strided/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
'decoder/layer_6/strided/strided_slice_3StridedSlicedecoder/layer_6/strided/stack-decoder/layer_6/strided/strided_slice_3/stack/decoder/layer_6/strided/strided_slice_3/stack_1/decoder/layer_6/strided/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
ъ
7decoder/layer_6/strided/conv2d_transpose/ReadVariableOpReadVariableOpdecoder/layer_6/strided/kernel*
dtype0*&
_output_shapes
: 
Ћ
(decoder/layer_6/strided/conv2d_transposeConv2DBackpropInputdecoder/layer_6/strided/stack7decoder/layer_6/strided/conv2d_transpose/ReadVariableOpdecoder/layer_5/ln/add_1*
T0*
strides
*1
_output_shapes
:         ђђ*
paddingSAME
Є
.decoder/layer_6/strided/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_6/strided/bias*
dtype0*
_output_shapes
:
└
decoder/layer_6/strided/BiasAddBiasAdd(decoder/layer_6/strided/conv2d_transpose.decoder/layer_6/strided/BiasAdd/ReadVariableOp*
T0*1
_output_shapes
:         ђђ
n
decoder/layer_6/strided/Shape_1Shapedecoder/layer_6/strided/BiasAdd*
T0*
_output_shapes
:
w
-decoder/layer_6/strided/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/decoder/layer_6/strided/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/decoder/layer_6/strided/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
»
'decoder/layer_6/strided/strided_slice_4StridedSlicedecoder/layer_6/strided/Shape_1-decoder/layer_6/strided/strided_slice_4/stack/decoder/layer_6/strided/strided_slice_4/stack_1/decoder/layer_6/strided/strided_slice_4/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
j
'decoder/layer_6/strided/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value
B :ђ
j
'decoder/layer_6/strided/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value
B :ђ
r
'decoder/layer_6/strided/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
i
'decoder/layer_6/strided/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
е
%decoder/layer_6/strided/Reshape/shapePack'decoder/layer_6/strided/strided_slice_4'decoder/layer_6/strided/Reshape/shape/1'decoder/layer_6/strided/Reshape/shape/2'decoder/layer_6/strided/Reshape/shape/3'decoder/layer_6/strided/Reshape/shape/4*
T0*
N*
_output_shapes
:
╗
decoder/layer_6/strided/ReshapeReshapedecoder/layer_6/strided/BiasAdd%decoder/layer_6/strided/Reshape/shape*
T0*>
_output_shapes,
*:(         ђђ         
_
decoder/layer_6/strided/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
r
'decoder/layer_6/strided/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
З
decoder/layer_6/strided/splitSplit'decoder/layer_6/strided/split/split_dimdecoder/layer_6/strided/Reshape*
T0*
	num_split*h
_output_shapesV
T:(         ђђ         :(         ђђ         
і
decoder/layer_6/strided/EluEludecoder/layer_6/strided/split*
T0*>
_output_shapes,
*:(         ђђ         
ї
decoder/layer_6/strided/NegNegdecoder/layer_6/strided/split:1*
T0*>
_output_shapes,
*:(         ђђ         
і
decoder/layer_6/strided/Elu_1Eludecoder/layer_6/strided/Neg*
T0*>
_output_shapes,
*:(         ђђ         
ї
decoder/layer_6/strided/Neg_1Negdecoder/layer_6/strided/Elu_1*
T0*>
_output_shapes,
*:(         ђђ         
n
#decoder/layer_6/strided/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
П
decoder/layer_6/strided/concatConcatV2decoder/layer_6/strided/Eludecoder/layer_6/strided/Neg_1#decoder/layer_6/strided/concat/axis*
N*>
_output_shapes,
*:(         ђђ         *
T0
l
)decoder/layer_6/strided/Reshape_1/shape/1Const*
value
B :ђ*
dtype0*
_output_shapes
: 
l
)decoder/layer_6/strided/Reshape_1/shape/2Const*
value
B :ђ*
dtype0*
_output_shapes
: 
k
)decoder/layer_6/strided/Reshape_1/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
Є
'decoder/layer_6/strided/Reshape_1/shapePack'decoder/layer_6/strided/strided_slice_4)decoder/layer_6/strided/Reshape_1/shape/1)decoder/layer_6/strided/Reshape_1/shape/2)decoder/layer_6/strided/Reshape_1/shape/3*
N*
_output_shapes
:*
T0
▒
!decoder/layer_6/strided/Reshape_1Reshapedecoder/layer_6/strided/concat'decoder/layer_6/strided/Reshape_1/shape*
T0*1
_output_shapes
:         ђђ
т
Ldecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*%
valueB"            
¤
Jdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*
valueB
 *?╚JЙ
¤
Jdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*
valueB
 *?╚J>*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:
╩
Jdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel
С
Jdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*&
_output_shapes
:
о
Fdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*&
_output_shapes
:
Щ
+decoder/layer_6/residual_0/depthwise_kernelVarHandleOp*
shape:*<
shared_name-+decoder/layer_6/residual_0/depthwise_kernel*>
_class4
20loc:@decoder/layer_6/residual_0/depthwise_kernel*
dtype0*
_output_shapes
: 
Д
Ldecoder/layer_6/residual_0/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_6/residual_0/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_6/residual_0/depthwise_kernel/AssignAssignVariableOp+decoder/layer_6/residual_0/depthwise_kernelFdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_6/residual_0/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_6/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:
т
Ldecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*%
valueB"             *
dtype0*
_output_shapes
:
¤
Jdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*
valueB
 *зхЙ*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*
valueB
 *зх>
▒
Tdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: 
╩
Jdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*&
_output_shapes
: 
о
Fdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*&
_output_shapes
: 
Щ
+decoder/layer_6/residual_0/pointwise_kernelVarHandleOp*<
shared_name-+decoder/layer_6/residual_0/pointwise_kernel*>
_class4
20loc:@decoder/layer_6/residual_0/pointwise_kernel*
dtype0*
_output_shapes
: *
shape: 
Д
Ldecoder/layer_6/residual_0/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_6/residual_0/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_6/residual_0/pointwise_kernel/AssignAssignVariableOp+decoder/layer_6/residual_0/pointwise_kernelFdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_6/residual_0/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_6/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: 
▓
1decoder/layer_6/residual_0/bias/Initializer/zerosConst*2
_class(
&$loc:@decoder/layer_6/residual_0/bias*
valueB *    *
dtype0*
_output_shapes
: 
╩
decoder/layer_6/residual_0/biasVarHandleOp*0
shared_name!decoder/layer_6/residual_0/bias*2
_class(
&$loc:@decoder/layer_6/residual_0/bias*
dtype0*
_output_shapes
: *
shape: 
Ј
@decoder/layer_6/residual_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_6/residual_0/bias*
_output_shapes
: 
Џ
&decoder/layer_6/residual_0/bias/AssignAssignVariableOpdecoder/layer_6/residual_0/bias1decoder/layer_6/residual_0/bias/Initializer/zeros*
dtype0
Ј
3decoder/layer_6/residual_0/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_6/residual_0/bias*
dtype0*
_output_shapes
: 
«
:decoder/layer_6/residual_0/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_6/residual_0/depthwise_kernel*
dtype0*&
_output_shapes
:
░
<decoder/layer_6/residual_0/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_6/residual_0/pointwise_kernel*
dtype0*&
_output_shapes
: 
і
1decoder/layer_6/residual_0/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
і
9decoder/layer_6/residual_0/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Љ
5decoder/layer_6/residual_0/separable_conv2d/depthwiseDepthwiseConv2dNative!decoder/layer_6/strided/Reshape_1:decoder/layer_6/residual_0/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*1
_output_shapes
:         ђђ
Ј
+decoder/layer_6/residual_0/separable_conv2dConv2D5decoder/layer_6/residual_0/separable_conv2d/depthwise<decoder/layer_6/residual_0/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*1
_output_shapes
:         ђђ 
Ї
1decoder/layer_6/residual_0/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_6/residual_0/bias*
dtype0*
_output_shapes
: 
╔
"decoder/layer_6/residual_0/BiasAddBiasAdd+decoder/layer_6/residual_0/separable_conv2d1decoder/layer_6/residual_0/BiasAdd/ReadVariableOp*1
_output_shapes
:         ђђ *
T0
r
 decoder/layer_6/residual_0/ShapeShape"decoder/layer_6/residual_0/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_6/residual_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_6/residual_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_6/residual_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_6/residual_0/strided_sliceStridedSlice decoder/layer_6/residual_0/Shape.decoder/layer_6/residual_0/strided_slice/stack0decoder/layer_6/residual_0/strided_slice/stack_10decoder/layer_6/residual_0/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
m
*decoder/layer_6/residual_0/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value
B :ђ
m
*decoder/layer_6/residual_0/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value
B :ђ
u
*decoder/layer_6/residual_0/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_6/residual_0/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_6/residual_0/Reshape/shapePack(decoder/layer_6/residual_0/strided_slice*decoder/layer_6/residual_0/Reshape/shape/1*decoder/layer_6/residual_0/Reshape/shape/2*decoder/layer_6/residual_0/Reshape/shape/3*decoder/layer_6/residual_0/Reshape/shape/4*
T0*
N*
_output_shapes
:
─
"decoder/layer_6/residual_0/ReshapeReshape"decoder/layer_6/residual_0/BiasAdd(decoder/layer_6/residual_0/Reshape/shape*
T0*>
_output_shapes,
*:(         ђђ         
b
 decoder/layer_6/residual_0/ConstConst*
dtype0*
_output_shapes
: *
value	B :
u
*decoder/layer_6/residual_0/split/split_dimConst*
dtype0*
_output_shapes
: *
valueB :
         
§
 decoder/layer_6/residual_0/splitSplit*decoder/layer_6/residual_0/split/split_dim"decoder/layer_6/residual_0/Reshape*
T0*
	num_split*h
_output_shapesV
T:(         ђђ         :(         ђђ         
љ
decoder/layer_6/residual_0/EluElu decoder/layer_6/residual_0/split*
T0*>
_output_shapes,
*:(         ђђ         
њ
decoder/layer_6/residual_0/NegNeg"decoder/layer_6/residual_0/split:1*
T0*>
_output_shapes,
*:(         ђђ         
љ
 decoder/layer_6/residual_0/Elu_1Eludecoder/layer_6/residual_0/Neg*
T0*>
_output_shapes,
*:(         ђђ         
њ
 decoder/layer_6/residual_0/Neg_1Neg decoder/layer_6/residual_0/Elu_1*
T0*>
_output_shapes,
*:(         ђђ         
q
&decoder/layer_6/residual_0/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
ж
!decoder/layer_6/residual_0/concatConcatV2decoder/layer_6/residual_0/Elu decoder/layer_6/residual_0/Neg_1&decoder/layer_6/residual_0/concat/axis*
T0*
N*>
_output_shapes,
*:(         ђђ         
o
,decoder/layer_6/residual_0/Reshape_1/shape/1Const*
value
B :ђ*
dtype0*
_output_shapes
: 
o
,decoder/layer_6/residual_0/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value
B :ђ
n
,decoder/layer_6/residual_0/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value	B : 
ћ
*decoder/layer_6/residual_0/Reshape_1/shapePack(decoder/layer_6/residual_0/strided_slice,decoder/layer_6/residual_0/Reshape_1/shape/1,decoder/layer_6/residual_0/Reshape_1/shape/2,decoder/layer_6/residual_0/Reshape_1/shape/3*
T0*
N*
_output_shapes
:
║
$decoder/layer_6/residual_0/Reshape_1Reshape!decoder/layer_6/residual_0/concat*decoder/layer_6/residual_0/Reshape_1/shape*
T0*1
_output_shapes
:         ђђ 
т
Ldecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel*%
valueB"             
¤
Jdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel*
valueB
 *ЄІЙ
¤
Jdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel*
valueB
 *ЄІ>*
dtype0*
_output_shapes
: 
▒
Tdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel
╩
Jdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/maxJdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel
С
Jdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel
о
Fdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniformAddJdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/mulJdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel*&
_output_shapes
: 
Щ
+decoder/layer_6/residual_1/depthwise_kernelVarHandleOp*>
_class4
20loc:@decoder/layer_6/residual_1/depthwise_kernel*
dtype0*
_output_shapes
: *
shape: *<
shared_name-+decoder/layer_6/residual_1/depthwise_kernel
Д
Ldecoder/layer_6/residual_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_6/residual_1/depthwise_kernel*
_output_shapes
: 
╚
2decoder/layer_6/residual_1/depthwise_kernel/AssignAssignVariableOp+decoder/layer_6/residual_1/depthwise_kernelFdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_6/residual_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_6/residual_1/depthwise_kernel*
dtype0*&
_output_shapes
: 
т
Ldecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*%
valueB"             
¤
Jdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*
valueB
 *зхЙ*
dtype0*
_output_shapes
: 
¤
Jdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*
valueB
 *зх>
▒
Tdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformLdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
: 
╩
Jdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/subSubJdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/maxJdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*
_output_shapes
: 
С
Jdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/mulMulTdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/RandomUniformJdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*&
_output_shapes
: 
о
Fdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniformAddJdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/mulJdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel*&
_output_shapes
: 
Щ
+decoder/layer_6/residual_1/pointwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *<
shared_name-+decoder/layer_6/residual_1/pointwise_kernel*>
_class4
20loc:@decoder/layer_6/residual_1/pointwise_kernel
Д
Ldecoder/layer_6/residual_1/pointwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+decoder/layer_6/residual_1/pointwise_kernel*
_output_shapes
: 
╚
2decoder/layer_6/residual_1/pointwise_kernel/AssignAssignVariableOp+decoder/layer_6/residual_1/pointwise_kernelFdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform*
dtype0
│
?decoder/layer_6/residual_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp+decoder/layer_6/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
: 
▓
1decoder/layer_6/residual_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@decoder/layer_6/residual_1/bias*
valueB*    
╩
decoder/layer_6/residual_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*0
shared_name!decoder/layer_6/residual_1/bias*2
_class(
&$loc:@decoder/layer_6/residual_1/bias
Ј
@decoder/layer_6/residual_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecoder/layer_6/residual_1/bias*
_output_shapes
: 
Џ
&decoder/layer_6/residual_1/bias/AssignAssignVariableOpdecoder/layer_6/residual_1/bias1decoder/layer_6/residual_1/bias/Initializer/zeros*
dtype0
Ј
3decoder/layer_6/residual_1/bias/Read/ReadVariableOpReadVariableOpdecoder/layer_6/residual_1/bias*
dtype0*
_output_shapes
:
«
:decoder/layer_6/residual_1/separable_conv2d/ReadVariableOpReadVariableOp+decoder/layer_6/residual_1/depthwise_kernel*
dtype0*&
_output_shapes
: 
░
<decoder/layer_6/residual_1/separable_conv2d/ReadVariableOp_1ReadVariableOp+decoder/layer_6/residual_1/pointwise_kernel*
dtype0*&
_output_shapes
: 
і
1decoder/layer_6/residual_1/separable_conv2d/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
і
9decoder/layer_6/residual_1/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ћ
5decoder/layer_6/residual_1/separable_conv2d/depthwiseDepthwiseConv2dNative$decoder/layer_6/residual_0/Reshape_1:decoder/layer_6/residual_1/separable_conv2d/ReadVariableOp*
paddingSAME*
T0*
strides
*1
_output_shapes
:         ђђ 
Ј
+decoder/layer_6/residual_1/separable_conv2dConv2D5decoder/layer_6/residual_1/separable_conv2d/depthwise<decoder/layer_6/residual_1/separable_conv2d/ReadVariableOp_1*
paddingVALID*
T0*
strides
*1
_output_shapes
:         ђђ
Ї
1decoder/layer_6/residual_1/BiasAdd/ReadVariableOpReadVariableOpdecoder/layer_6/residual_1/bias*
dtype0*
_output_shapes
:
╔
"decoder/layer_6/residual_1/BiasAddBiasAdd+decoder/layer_6/residual_1/separable_conv2d1decoder/layer_6/residual_1/BiasAdd/ReadVariableOp*
T0*1
_output_shapes
:         ђђ
r
 decoder/layer_6/residual_1/ShapeShape"decoder/layer_6/residual_1/BiasAdd*
T0*
_output_shapes
:
x
.decoder/layer_6/residual_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0decoder/layer_6/residual_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0decoder/layer_6/residual_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┤
(decoder/layer_6/residual_1/strided_sliceStridedSlice decoder/layer_6/residual_1/Shape.decoder/layer_6/residual_1/strided_slice/stack0decoder/layer_6/residual_1/strided_slice/stack_10decoder/layer_6/residual_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
m
*decoder/layer_6/residual_1/Reshape/shape/1Const*
value
B :ђ*
dtype0*
_output_shapes
: 
m
*decoder/layer_6/residual_1/Reshape/shape/2Const*
value
B :ђ*
dtype0*
_output_shapes
: 
u
*decoder/layer_6/residual_1/Reshape/shape/3Const*
valueB :
         *
dtype0*
_output_shapes
: 
l
*decoder/layer_6/residual_1/Reshape/shape/4Const*
value	B :*
dtype0*
_output_shapes
: 
И
(decoder/layer_6/residual_1/Reshape/shapePack(decoder/layer_6/residual_1/strided_slice*decoder/layer_6/residual_1/Reshape/shape/1*decoder/layer_6/residual_1/Reshape/shape/2*decoder/layer_6/residual_1/Reshape/shape/3*decoder/layer_6/residual_1/Reshape/shape/4*
N*
_output_shapes
:*
T0
─
"decoder/layer_6/residual_1/ReshapeReshape"decoder/layer_6/residual_1/BiasAdd(decoder/layer_6/residual_1/Reshape/shape*
T0*>
_output_shapes,
*:(         ђђ         
b
 decoder/layer_6/residual_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
u
*decoder/layer_6/residual_1/split/split_dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
§
 decoder/layer_6/residual_1/splitSplit*decoder/layer_6/residual_1/split/split_dim"decoder/layer_6/residual_1/Reshape*
	num_split*h
_output_shapesV
T:(         ђђ         :(         ђђ         *
T0
љ
decoder/layer_6/residual_1/EluElu decoder/layer_6/residual_1/split*>
_output_shapes,
*:(         ђђ         *
T0
њ
decoder/layer_6/residual_1/NegNeg"decoder/layer_6/residual_1/split:1*
T0*>
_output_shapes,
*:(         ђђ         
љ
 decoder/layer_6/residual_1/Elu_1Eludecoder/layer_6/residual_1/Neg*
T0*>
_output_shapes,
*:(         ђђ         
њ
 decoder/layer_6/residual_1/Neg_1Neg decoder/layer_6/residual_1/Elu_1*
T0*>
_output_shapes,
*:(         ђђ         
q
&decoder/layer_6/residual_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
ж
!decoder/layer_6/residual_1/concatConcatV2decoder/layer_6/residual_1/Elu decoder/layer_6/residual_1/Neg_1&decoder/layer_6/residual_1/concat/axis*
N*>
_output_shapes,
*:(         ђђ         *
T0
o
,decoder/layer_6/residual_1/Reshape_1/shape/1Const*
value
B :ђ*
dtype0*
_output_shapes
: 
o
,decoder/layer_6/residual_1/Reshape_1/shape/2Const*
value
B :ђ*
dtype0*
_output_shapes
: 
n
,decoder/layer_6/residual_1/Reshape_1/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
ћ
*decoder/layer_6/residual_1/Reshape_1/shapePack(decoder/layer_6/residual_1/strided_slice,decoder/layer_6/residual_1/Reshape_1/shape/1,decoder/layer_6/residual_1/Reshape_1/shape/2,decoder/layer_6/residual_1/Reshape_1/shape/3*
N*
_output_shapes
:*
T0
║
$decoder/layer_6/residual_1/Reshape_1Reshape!decoder/layer_6/residual_1/concat*decoder/layer_6/residual_1/Reshape_1/shape*1
_output_shapes
:         ђђ*
T0
А
decoder/layer_6/addAddV2!decoder/layer_6/strided/Reshape_1$decoder/layer_6/residual_1/Reshape_1*
T0*1
_output_shapes
:         ђђ
X
decoder/layer_6/ShapeShapedecoder/layer_6/add*
_output_shapes
:*
T0
m
#decoder/layer_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
decoder/layer_6/strided_sliceStridedSlicedecoder/layer_6/Shape#decoder/layer_6/strided_slice/stack%decoder/layer_6/strided_slice/stack_1%decoder/layer_6/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
╣
4decoder/layer_6/ln/layer_norm_scale/Initializer/onesConst*6
_class,
*(loc:@decoder/layer_6/ln/layer_norm_scale*
valueB*  ђ?*
dtype0*
_output_shapes
:
о
#decoder/layer_6/ln/layer_norm_scaleVarHandleOp*6
_class,
*(loc:@decoder/layer_6/ln/layer_norm_scale*
dtype0*
_output_shapes
: *
shape:*4
shared_name%#decoder/layer_6/ln/layer_norm_scale
Ќ
Ddecoder/layer_6/ln/layer_norm_scale/IsInitialized/VarIsInitializedOpVarIsInitializedOp#decoder/layer_6/ln/layer_norm_scale*
_output_shapes
: 
д
*decoder/layer_6/ln/layer_norm_scale/AssignAssignVariableOp#decoder/layer_6/ln/layer_norm_scale4decoder/layer_6/ln/layer_norm_scale/Initializer/ones*
dtype0
Ќ
7decoder/layer_6/ln/layer_norm_scale/Read/ReadVariableOpReadVariableOp#decoder/layer_6/ln/layer_norm_scale*
dtype0*
_output_shapes
:
И
4decoder/layer_6/ln/layer_norm_bias/Initializer/zerosConst*5
_class+
)'loc:@decoder/layer_6/ln/layer_norm_bias*
valueB*    *
dtype0*
_output_shapes
:
М
"decoder/layer_6/ln/layer_norm_biasVarHandleOp*
shape:*3
shared_name$"decoder/layer_6/ln/layer_norm_bias*5
_class+
)'loc:@decoder/layer_6/ln/layer_norm_bias*
dtype0*
_output_shapes
: 
Ћ
Cdecoder/layer_6/ln/layer_norm_bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"decoder/layer_6/ln/layer_norm_bias*
_output_shapes
: 
ц
)decoder/layer_6/ln/layer_norm_bias/AssignAssignVariableOp"decoder/layer_6/ln/layer_norm_bias4decoder/layer_6/ln/layer_norm_bias/Initializer/zeros*
dtype0
Ћ
6decoder/layer_6/ln/layer_norm_bias/Read/ReadVariableOpReadVariableOp"decoder/layer_6/ln/layer_norm_bias*
dtype0*
_output_shapes
:
]
decoder/layer_6/ln/ConstConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ђ
!decoder/layer_6/ln/ReadVariableOpReadVariableOp#decoder/layer_6/ln/layer_norm_scale*
dtype0*
_output_shapes
:
ѓ
#decoder/layer_6/ln/ReadVariableOp_1ReadVariableOp"decoder/layer_6/ln/layer_norm_bias*
dtype0*
_output_shapes
:
|
)decoder/layer_6/ln/Mean/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
г
decoder/layer_6/ln/MeanMeandecoder/layer_6/add)decoder/layer_6/ln/Mean/reduction_indices*1
_output_shapes
:         ђђ*
	keep_dims(*
T0
Б
$decoder/layer_6/ln/SquaredDifferenceSquaredDifferencedecoder/layer_6/adddecoder/layer_6/ln/Mean*
T0*1
_output_shapes
:         ђђ
~
+decoder/layer_6/ln/Mean_1/reduction_indicesConst*
valueB:
         *
dtype0*
_output_shapes
:
┴
decoder/layer_6/ln/Mean_1Mean$decoder/layer_6/ln/SquaredDifference+decoder/layer_6/ln/Mean_1/reduction_indices*
T0*1
_output_shapes
:         ђђ*
	keep_dims(
Є
decoder/layer_6/ln/subSubdecoder/layer_6/adddecoder/layer_6/ln/Mean*
T0*1
_output_shapes
:         ђђ
љ
decoder/layer_6/ln/addAddV2decoder/layer_6/ln/Mean_1decoder/layer_6/ln/Const*
T0*1
_output_shapes
:         ђђ
u
decoder/layer_6/ln/RsqrtRsqrtdecoder/layer_6/ln/add*
T0*1
_output_shapes
:         ђђ
І
decoder/layer_6/ln/mulMuldecoder/layer_6/ln/subdecoder/layer_6/ln/Rsqrt*
T0*1
_output_shapes
:         ђђ
ќ
decoder/layer_6/ln/mul_1Muldecoder/layer_6/ln/mul!decoder/layer_6/ln/ReadVariableOp*
T0*1
_output_shapes
:         ђђ
ю
decoder/layer_6/ln/add_1AddV2decoder/layer_6/ln/mul_1#decoder/layer_6/ln/ReadVariableOp_1*
T0*1
_output_shapes
:         ђђ
и
9autoencoder_final/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@autoencoder_final/kernel*
valueB"      
Е
7autoencoder_final/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *+
_class!
loc:@autoencoder_final/kernel*
valueB
 *0┐
Е
7autoencoder_final/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@autoencoder_final/kernel*
valueB
 *0?*
dtype0*
_output_shapes
: 
­
Aautoencoder_final/kernel/Initializer/random_uniform/RandomUniformRandomUniform9autoencoder_final/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@autoencoder_final/kernel*
dtype0*
_output_shapes

:
■
7autoencoder_final/kernel/Initializer/random_uniform/subSub7autoencoder_final/kernel/Initializer/random_uniform/max7autoencoder_final/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@autoencoder_final/kernel*
_output_shapes
: 
љ
7autoencoder_final/kernel/Initializer/random_uniform/mulMulAautoencoder_final/kernel/Initializer/random_uniform/RandomUniform7autoencoder_final/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@autoencoder_final/kernel*
_output_shapes

:
ѓ
3autoencoder_final/kernel/Initializer/random_uniformAdd7autoencoder_final/kernel/Initializer/random_uniform/mul7autoencoder_final/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@autoencoder_final/kernel*
_output_shapes

:
╣
autoencoder_final/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*)
shared_nameautoencoder_final/kernel*+
_class!
loc:@autoencoder_final/kernel
Ђ
9autoencoder_final/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpautoencoder_final/kernel*
_output_shapes
: 
Ј
autoencoder_final/kernel/AssignAssignVariableOpautoencoder_final/kernel3autoencoder_final/kernel/Initializer/random_uniform*
dtype0
Ё
,autoencoder_final/kernel/Read/ReadVariableOpReadVariableOpautoencoder_final/kernel*
dtype0*
_output_shapes

:
а
(autoencoder_final/bias/Initializer/zerosConst*)
_class
loc:@autoencoder_final/bias*
valueB*    *
dtype0*
_output_shapes
:
»
autoencoder_final/biasVarHandleOp*)
_class
loc:@autoencoder_final/bias*
dtype0*
_output_shapes
: *
shape:*'
shared_nameautoencoder_final/bias
}
7autoencoder_final/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpautoencoder_final/bias*
_output_shapes
: 
ђ
autoencoder_final/bias/AssignAssignVariableOpautoencoder_final/bias(autoencoder_final/bias/Initializer/zeros*
dtype0
}
*autoencoder_final/bias/Read/ReadVariableOpReadVariableOpautoencoder_final/bias*
dtype0*
_output_shapes
:
Ѓ
*autoencoder_final/Tensordot/ReadVariableOpReadVariableOpautoencoder_final/kernel*
dtype0*
_output_shapes

:
j
 autoencoder_final/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:
u
 autoencoder_final/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          
i
!autoencoder_final/Tensordot/ShapeShapedecoder/layer_6/ln/add_1*
_output_shapes
:*
T0
k
)autoencoder_final/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
У
$autoencoder_final/Tensordot/GatherV2GatherV2!autoencoder_final/Tensordot/Shape autoencoder_final/Tensordot/free)autoencoder_final/Tensordot/GatherV2/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
m
+autoencoder_final/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
В
&autoencoder_final/Tensordot/GatherV2_1GatherV2!autoencoder_final/Tensordot/Shape autoencoder_final/Tensordot/axes+autoencoder_final/Tensordot/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
k
!autoencoder_final/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
њ
 autoencoder_final/Tensordot/ProdProd$autoencoder_final/Tensordot/GatherV2!autoencoder_final/Tensordot/Const*
_output_shapes
: *
T0
m
#autoencoder_final/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
ў
"autoencoder_final/Tensordot/Prod_1Prod&autoencoder_final/Tensordot/GatherV2_1#autoencoder_final/Tensordot/Const_1*
T0*
_output_shapes
: 
i
'autoencoder_final/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╔
"autoencoder_final/Tensordot/concatConcatV2 autoencoder_final/Tensordot/free autoencoder_final/Tensordot/axes'autoencoder_final/Tensordot/concat/axis*
N*
_output_shapes
:*
T0
Ю
!autoencoder_final/Tensordot/stackPack autoencoder_final/Tensordot/Prod"autoencoder_final/Tensordot/Prod_1*
T0*
N*
_output_shapes
:
г
%autoencoder_final/Tensordot/transpose	Transposedecoder/layer_6/ln/add_1"autoencoder_final/Tensordot/concat*1
_output_shapes
:         ђђ*
T0
│
#autoencoder_final/Tensordot/ReshapeReshape%autoencoder_final/Tensordot/transpose!autoencoder_final/Tensordot/stack*
T0*0
_output_shapes
:                  
}
,autoencoder_final/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
и
'autoencoder_final/Tensordot/transpose_1	Transpose*autoencoder_final/Tensordot/ReadVariableOp,autoencoder_final/Tensordot/transpose_1/perm*
T0*
_output_shapes

:
|
+autoencoder_final/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
»
%autoencoder_final/Tensordot/Reshape_1Reshape'autoencoder_final/Tensordot/transpose_1+autoencoder_final/Tensordot/Reshape_1/shape*
T0*
_output_shapes

:
ф
"autoencoder_final/Tensordot/MatMulMatMul#autoencoder_final/Tensordot/Reshape%autoencoder_final/Tensordot/Reshape_1*
T0*'
_output_shapes
:         
m
#autoencoder_final/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
k
)autoencoder_final/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
н
$autoencoder_final/Tensordot/concat_1ConcatV2$autoencoder_final/Tensordot/GatherV2#autoencoder_final/Tensordot/Const_2)autoencoder_final/Tensordot/concat_1/axis*
T0*
N*
_output_shapes
:
г
autoencoder_final/TensordotReshape"autoencoder_final/Tensordot/MatMul$autoencoder_final/Tensordot/concat_1*
T0*1
_output_shapes
:         ђђ
{
(autoencoder_final/BiasAdd/ReadVariableOpReadVariableOpautoencoder_final/bias*
dtype0*
_output_shapes
:
Д
autoencoder_final/BiasAddBiasAddautoencoder_final/Tensordot(autoencoder_final/BiasAdd/ReadVariableOp*
T0*1
_output_shapes
:         ђђ
}
autoencoder_final/SoftplusSoftplusautoencoder_final/BiasAdd*
T0*1
_output_shapes
:         ђђ"є"жz
trainable_variablesЛz╬z
г
unbottleneck/dense/kernel:0 unbottleneck/dense/kernel/Assign/unbottleneck/dense/kernel/Read/ReadVariableOp:0(26unbottleneck/dense/kernel/Initializer/random_uniform:08
Џ
unbottleneck/dense/bias:0unbottleneck/dense/bias/Assign-unbottleneck/dense/bias/Read/ReadVariableOp:0(2+unbottleneck/dense/bias/Initializer/zeros:08
└
 decoder/layer_0/strided/kernel:0%decoder/layer_0/strided/kernel/Assign4decoder/layer_0/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_0/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_0/strided/bias:0#decoder/layer_0/strided/bias/Assign2decoder/layer_0/strided/bias/Read/ReadVariableOp:0(20decoder/layer_0/strided/bias/Initializer/zeros:08
З
-decoder/layer_0/residual_0/depthwise_kernel:02decoder/layer_0/residual_0/depthwise_kernel/AssignAdecoder/layer_0/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_0/residual_0/pointwise_kernel:02decoder/layer_0/residual_0/pointwise_kernel/AssignAdecoder/layer_0/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_0/residual_0/bias:0&decoder/layer_0/residual_0/bias/Assign5decoder/layer_0/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_0/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_0/residual_1/depthwise_kernel:02decoder/layer_0/residual_1/depthwise_kernel/AssignAdecoder/layer_0/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_0/residual_1/pointwise_kernel:02decoder/layer_0/residual_1/pointwise_kernel/AssignAdecoder/layer_0/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_0/residual_1/bias:0&decoder/layer_0/residual_1/bias/Assign5decoder/layer_0/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_0/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_0/ln/layer_norm_scale:0*decoder/layer_0/ln/layer_norm_scale/Assign9decoder/layer_0/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_0/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_0/ln/layer_norm_bias:0)decoder/layer_0/ln/layer_norm_bias/Assign8decoder/layer_0/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_0/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_1/strided/kernel:0%decoder/layer_1/strided/kernel/Assign4decoder/layer_1/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_1/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_1/strided/bias:0#decoder/layer_1/strided/bias/Assign2decoder/layer_1/strided/bias/Read/ReadVariableOp:0(20decoder/layer_1/strided/bias/Initializer/zeros:08
З
-decoder/layer_1/residual_0/depthwise_kernel:02decoder/layer_1/residual_0/depthwise_kernel/AssignAdecoder/layer_1/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_1/residual_0/pointwise_kernel:02decoder/layer_1/residual_0/pointwise_kernel/AssignAdecoder/layer_1/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_1/residual_0/bias:0&decoder/layer_1/residual_0/bias/Assign5decoder/layer_1/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_1/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_1/residual_1/depthwise_kernel:02decoder/layer_1/residual_1/depthwise_kernel/AssignAdecoder/layer_1/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_1/residual_1/pointwise_kernel:02decoder/layer_1/residual_1/pointwise_kernel/AssignAdecoder/layer_1/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_1/residual_1/bias:0&decoder/layer_1/residual_1/bias/Assign5decoder/layer_1/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_1/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_1/ln/layer_norm_scale:0*decoder/layer_1/ln/layer_norm_scale/Assign9decoder/layer_1/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_1/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_1/ln/layer_norm_bias:0)decoder/layer_1/ln/layer_norm_bias/Assign8decoder/layer_1/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_1/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_2/strided/kernel:0%decoder/layer_2/strided/kernel/Assign4decoder/layer_2/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_2/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_2/strided/bias:0#decoder/layer_2/strided/bias/Assign2decoder/layer_2/strided/bias/Read/ReadVariableOp:0(20decoder/layer_2/strided/bias/Initializer/zeros:08
З
-decoder/layer_2/residual_0/depthwise_kernel:02decoder/layer_2/residual_0/depthwise_kernel/AssignAdecoder/layer_2/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_2/residual_0/pointwise_kernel:02decoder/layer_2/residual_0/pointwise_kernel/AssignAdecoder/layer_2/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_2/residual_0/bias:0&decoder/layer_2/residual_0/bias/Assign5decoder/layer_2/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_2/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_2/residual_1/depthwise_kernel:02decoder/layer_2/residual_1/depthwise_kernel/AssignAdecoder/layer_2/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_2/residual_1/pointwise_kernel:02decoder/layer_2/residual_1/pointwise_kernel/AssignAdecoder/layer_2/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_2/residual_1/bias:0&decoder/layer_2/residual_1/bias/Assign5decoder/layer_2/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_2/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_2/ln/layer_norm_scale:0*decoder/layer_2/ln/layer_norm_scale/Assign9decoder/layer_2/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_2/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_2/ln/layer_norm_bias:0)decoder/layer_2/ln/layer_norm_bias/Assign8decoder/layer_2/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_2/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_3/strided/kernel:0%decoder/layer_3/strided/kernel/Assign4decoder/layer_3/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_3/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_3/strided/bias:0#decoder/layer_3/strided/bias/Assign2decoder/layer_3/strided/bias/Read/ReadVariableOp:0(20decoder/layer_3/strided/bias/Initializer/zeros:08
З
-decoder/layer_3/residual_0/depthwise_kernel:02decoder/layer_3/residual_0/depthwise_kernel/AssignAdecoder/layer_3/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_3/residual_0/pointwise_kernel:02decoder/layer_3/residual_0/pointwise_kernel/AssignAdecoder/layer_3/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_3/residual_0/bias:0&decoder/layer_3/residual_0/bias/Assign5decoder/layer_3/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_3/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_3/residual_1/depthwise_kernel:02decoder/layer_3/residual_1/depthwise_kernel/AssignAdecoder/layer_3/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_3/residual_1/pointwise_kernel:02decoder/layer_3/residual_1/pointwise_kernel/AssignAdecoder/layer_3/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_3/residual_1/bias:0&decoder/layer_3/residual_1/bias/Assign5decoder/layer_3/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_3/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_3/ln/layer_norm_scale:0*decoder/layer_3/ln/layer_norm_scale/Assign9decoder/layer_3/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_3/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_3/ln/layer_norm_bias:0)decoder/layer_3/ln/layer_norm_bias/Assign8decoder/layer_3/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_3/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_4/strided/kernel:0%decoder/layer_4/strided/kernel/Assign4decoder/layer_4/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_4/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_4/strided/bias:0#decoder/layer_4/strided/bias/Assign2decoder/layer_4/strided/bias/Read/ReadVariableOp:0(20decoder/layer_4/strided/bias/Initializer/zeros:08
З
-decoder/layer_4/residual_0/depthwise_kernel:02decoder/layer_4/residual_0/depthwise_kernel/AssignAdecoder/layer_4/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_4/residual_0/pointwise_kernel:02decoder/layer_4/residual_0/pointwise_kernel/AssignAdecoder/layer_4/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_4/residual_0/bias:0&decoder/layer_4/residual_0/bias/Assign5decoder/layer_4/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_4/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_4/residual_1/depthwise_kernel:02decoder/layer_4/residual_1/depthwise_kernel/AssignAdecoder/layer_4/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_4/residual_1/pointwise_kernel:02decoder/layer_4/residual_1/pointwise_kernel/AssignAdecoder/layer_4/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_4/residual_1/bias:0&decoder/layer_4/residual_1/bias/Assign5decoder/layer_4/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_4/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_4/ln/layer_norm_scale:0*decoder/layer_4/ln/layer_norm_scale/Assign9decoder/layer_4/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_4/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_4/ln/layer_norm_bias:0)decoder/layer_4/ln/layer_norm_bias/Assign8decoder/layer_4/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_4/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_5/strided/kernel:0%decoder/layer_5/strided/kernel/Assign4decoder/layer_5/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_5/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_5/strided/bias:0#decoder/layer_5/strided/bias/Assign2decoder/layer_5/strided/bias/Read/ReadVariableOp:0(20decoder/layer_5/strided/bias/Initializer/zeros:08
З
-decoder/layer_5/residual_0/depthwise_kernel:02decoder/layer_5/residual_0/depthwise_kernel/AssignAdecoder/layer_5/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_5/residual_0/pointwise_kernel:02decoder/layer_5/residual_0/pointwise_kernel/AssignAdecoder/layer_5/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_5/residual_0/bias:0&decoder/layer_5/residual_0/bias/Assign5decoder/layer_5/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_5/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_5/residual_1/depthwise_kernel:02decoder/layer_5/residual_1/depthwise_kernel/AssignAdecoder/layer_5/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_5/residual_1/pointwise_kernel:02decoder/layer_5/residual_1/pointwise_kernel/AssignAdecoder/layer_5/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_5/residual_1/bias:0&decoder/layer_5/residual_1/bias/Assign5decoder/layer_5/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_5/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_5/ln/layer_norm_scale:0*decoder/layer_5/ln/layer_norm_scale/Assign9decoder/layer_5/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_5/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_5/ln/layer_norm_bias:0)decoder/layer_5/ln/layer_norm_bias/Assign8decoder/layer_5/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_5/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_6/strided/kernel:0%decoder/layer_6/strided/kernel/Assign4decoder/layer_6/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_6/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_6/strided/bias:0#decoder/layer_6/strided/bias/Assign2decoder/layer_6/strided/bias/Read/ReadVariableOp:0(20decoder/layer_6/strided/bias/Initializer/zeros:08
З
-decoder/layer_6/residual_0/depthwise_kernel:02decoder/layer_6/residual_0/depthwise_kernel/AssignAdecoder/layer_6/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_6/residual_0/pointwise_kernel:02decoder/layer_6/residual_0/pointwise_kernel/AssignAdecoder/layer_6/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_6/residual_0/bias:0&decoder/layer_6/residual_0/bias/Assign5decoder/layer_6/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_6/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_6/residual_1/depthwise_kernel:02decoder/layer_6/residual_1/depthwise_kernel/AssignAdecoder/layer_6/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_6/residual_1/pointwise_kernel:02decoder/layer_6/residual_1/pointwise_kernel/AssignAdecoder/layer_6/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_6/residual_1/bias:0&decoder/layer_6/residual_1/bias/Assign5decoder/layer_6/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_6/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_6/ln/layer_norm_scale:0*decoder/layer_6/ln/layer_norm_scale/Assign9decoder/layer_6/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_6/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_6/ln/layer_norm_bias:0)decoder/layer_6/ln/layer_norm_bias/Assign8decoder/layer_6/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_6/ln/layer_norm_bias/Initializer/zeros:08
е
autoencoder_final/kernel:0autoencoder_final/kernel/Assign.autoencoder_final/kernel/Read/ReadVariableOp:0(25autoencoder_final/kernel/Initializer/random_uniform:08
Ќ
autoencoder_final/bias:0autoencoder_final/bias/Assign,autoencoder_final/bias/Read/ReadVariableOp:0(2*autoencoder_final/bias/Initializer/zeros:08"▀z
	variablesЛz╬z
г
unbottleneck/dense/kernel:0 unbottleneck/dense/kernel/Assign/unbottleneck/dense/kernel/Read/ReadVariableOp:0(26unbottleneck/dense/kernel/Initializer/random_uniform:08
Џ
unbottleneck/dense/bias:0unbottleneck/dense/bias/Assign-unbottleneck/dense/bias/Read/ReadVariableOp:0(2+unbottleneck/dense/bias/Initializer/zeros:08
└
 decoder/layer_0/strided/kernel:0%decoder/layer_0/strided/kernel/Assign4decoder/layer_0/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_0/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_0/strided/bias:0#decoder/layer_0/strided/bias/Assign2decoder/layer_0/strided/bias/Read/ReadVariableOp:0(20decoder/layer_0/strided/bias/Initializer/zeros:08
З
-decoder/layer_0/residual_0/depthwise_kernel:02decoder/layer_0/residual_0/depthwise_kernel/AssignAdecoder/layer_0/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_0/residual_0/pointwise_kernel:02decoder/layer_0/residual_0/pointwise_kernel/AssignAdecoder/layer_0/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_0/residual_0/bias:0&decoder/layer_0/residual_0/bias/Assign5decoder/layer_0/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_0/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_0/residual_1/depthwise_kernel:02decoder/layer_0/residual_1/depthwise_kernel/AssignAdecoder/layer_0/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_0/residual_1/pointwise_kernel:02decoder/layer_0/residual_1/pointwise_kernel/AssignAdecoder/layer_0/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_0/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_0/residual_1/bias:0&decoder/layer_0/residual_1/bias/Assign5decoder/layer_0/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_0/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_0/ln/layer_norm_scale:0*decoder/layer_0/ln/layer_norm_scale/Assign9decoder/layer_0/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_0/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_0/ln/layer_norm_bias:0)decoder/layer_0/ln/layer_norm_bias/Assign8decoder/layer_0/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_0/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_1/strided/kernel:0%decoder/layer_1/strided/kernel/Assign4decoder/layer_1/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_1/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_1/strided/bias:0#decoder/layer_1/strided/bias/Assign2decoder/layer_1/strided/bias/Read/ReadVariableOp:0(20decoder/layer_1/strided/bias/Initializer/zeros:08
З
-decoder/layer_1/residual_0/depthwise_kernel:02decoder/layer_1/residual_0/depthwise_kernel/AssignAdecoder/layer_1/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_1/residual_0/pointwise_kernel:02decoder/layer_1/residual_0/pointwise_kernel/AssignAdecoder/layer_1/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_1/residual_0/bias:0&decoder/layer_1/residual_0/bias/Assign5decoder/layer_1/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_1/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_1/residual_1/depthwise_kernel:02decoder/layer_1/residual_1/depthwise_kernel/AssignAdecoder/layer_1/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_1/residual_1/pointwise_kernel:02decoder/layer_1/residual_1/pointwise_kernel/AssignAdecoder/layer_1/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_1/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_1/residual_1/bias:0&decoder/layer_1/residual_1/bias/Assign5decoder/layer_1/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_1/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_1/ln/layer_norm_scale:0*decoder/layer_1/ln/layer_norm_scale/Assign9decoder/layer_1/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_1/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_1/ln/layer_norm_bias:0)decoder/layer_1/ln/layer_norm_bias/Assign8decoder/layer_1/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_1/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_2/strided/kernel:0%decoder/layer_2/strided/kernel/Assign4decoder/layer_2/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_2/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_2/strided/bias:0#decoder/layer_2/strided/bias/Assign2decoder/layer_2/strided/bias/Read/ReadVariableOp:0(20decoder/layer_2/strided/bias/Initializer/zeros:08
З
-decoder/layer_2/residual_0/depthwise_kernel:02decoder/layer_2/residual_0/depthwise_kernel/AssignAdecoder/layer_2/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_2/residual_0/pointwise_kernel:02decoder/layer_2/residual_0/pointwise_kernel/AssignAdecoder/layer_2/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_2/residual_0/bias:0&decoder/layer_2/residual_0/bias/Assign5decoder/layer_2/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_2/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_2/residual_1/depthwise_kernel:02decoder/layer_2/residual_1/depthwise_kernel/AssignAdecoder/layer_2/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_2/residual_1/pointwise_kernel:02decoder/layer_2/residual_1/pointwise_kernel/AssignAdecoder/layer_2/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_2/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_2/residual_1/bias:0&decoder/layer_2/residual_1/bias/Assign5decoder/layer_2/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_2/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_2/ln/layer_norm_scale:0*decoder/layer_2/ln/layer_norm_scale/Assign9decoder/layer_2/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_2/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_2/ln/layer_norm_bias:0)decoder/layer_2/ln/layer_norm_bias/Assign8decoder/layer_2/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_2/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_3/strided/kernel:0%decoder/layer_3/strided/kernel/Assign4decoder/layer_3/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_3/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_3/strided/bias:0#decoder/layer_3/strided/bias/Assign2decoder/layer_3/strided/bias/Read/ReadVariableOp:0(20decoder/layer_3/strided/bias/Initializer/zeros:08
З
-decoder/layer_3/residual_0/depthwise_kernel:02decoder/layer_3/residual_0/depthwise_kernel/AssignAdecoder/layer_3/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_3/residual_0/pointwise_kernel:02decoder/layer_3/residual_0/pointwise_kernel/AssignAdecoder/layer_3/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_3/residual_0/bias:0&decoder/layer_3/residual_0/bias/Assign5decoder/layer_3/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_3/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_3/residual_1/depthwise_kernel:02decoder/layer_3/residual_1/depthwise_kernel/AssignAdecoder/layer_3/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_3/residual_1/pointwise_kernel:02decoder/layer_3/residual_1/pointwise_kernel/AssignAdecoder/layer_3/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_3/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_3/residual_1/bias:0&decoder/layer_3/residual_1/bias/Assign5decoder/layer_3/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_3/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_3/ln/layer_norm_scale:0*decoder/layer_3/ln/layer_norm_scale/Assign9decoder/layer_3/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_3/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_3/ln/layer_norm_bias:0)decoder/layer_3/ln/layer_norm_bias/Assign8decoder/layer_3/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_3/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_4/strided/kernel:0%decoder/layer_4/strided/kernel/Assign4decoder/layer_4/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_4/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_4/strided/bias:0#decoder/layer_4/strided/bias/Assign2decoder/layer_4/strided/bias/Read/ReadVariableOp:0(20decoder/layer_4/strided/bias/Initializer/zeros:08
З
-decoder/layer_4/residual_0/depthwise_kernel:02decoder/layer_4/residual_0/depthwise_kernel/AssignAdecoder/layer_4/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_4/residual_0/pointwise_kernel:02decoder/layer_4/residual_0/pointwise_kernel/AssignAdecoder/layer_4/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_4/residual_0/bias:0&decoder/layer_4/residual_0/bias/Assign5decoder/layer_4/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_4/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_4/residual_1/depthwise_kernel:02decoder/layer_4/residual_1/depthwise_kernel/AssignAdecoder/layer_4/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_4/residual_1/pointwise_kernel:02decoder/layer_4/residual_1/pointwise_kernel/AssignAdecoder/layer_4/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_4/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_4/residual_1/bias:0&decoder/layer_4/residual_1/bias/Assign5decoder/layer_4/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_4/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_4/ln/layer_norm_scale:0*decoder/layer_4/ln/layer_norm_scale/Assign9decoder/layer_4/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_4/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_4/ln/layer_norm_bias:0)decoder/layer_4/ln/layer_norm_bias/Assign8decoder/layer_4/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_4/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_5/strided/kernel:0%decoder/layer_5/strided/kernel/Assign4decoder/layer_5/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_5/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_5/strided/bias:0#decoder/layer_5/strided/bias/Assign2decoder/layer_5/strided/bias/Read/ReadVariableOp:0(20decoder/layer_5/strided/bias/Initializer/zeros:08
З
-decoder/layer_5/residual_0/depthwise_kernel:02decoder/layer_5/residual_0/depthwise_kernel/AssignAdecoder/layer_5/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_5/residual_0/pointwise_kernel:02decoder/layer_5/residual_0/pointwise_kernel/AssignAdecoder/layer_5/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_5/residual_0/bias:0&decoder/layer_5/residual_0/bias/Assign5decoder/layer_5/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_5/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_5/residual_1/depthwise_kernel:02decoder/layer_5/residual_1/depthwise_kernel/AssignAdecoder/layer_5/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_5/residual_1/pointwise_kernel:02decoder/layer_5/residual_1/pointwise_kernel/AssignAdecoder/layer_5/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_5/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_5/residual_1/bias:0&decoder/layer_5/residual_1/bias/Assign5decoder/layer_5/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_5/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_5/ln/layer_norm_scale:0*decoder/layer_5/ln/layer_norm_scale/Assign9decoder/layer_5/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_5/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_5/ln/layer_norm_bias:0)decoder/layer_5/ln/layer_norm_bias/Assign8decoder/layer_5/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_5/ln/layer_norm_bias/Initializer/zeros:08
└
 decoder/layer_6/strided/kernel:0%decoder/layer_6/strided/kernel/Assign4decoder/layer_6/strided/kernel/Read/ReadVariableOp:0(2;decoder/layer_6/strided/kernel/Initializer/random_uniform:08
»
decoder/layer_6/strided/bias:0#decoder/layer_6/strided/bias/Assign2decoder/layer_6/strided/bias/Read/ReadVariableOp:0(20decoder/layer_6/strided/bias/Initializer/zeros:08
З
-decoder/layer_6/residual_0/depthwise_kernel:02decoder/layer_6/residual_0/depthwise_kernel/AssignAdecoder/layer_6/residual_0/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_0/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_6/residual_0/pointwise_kernel:02decoder/layer_6/residual_0/pointwise_kernel/AssignAdecoder/layer_6/residual_0/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_0/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_6/residual_0/bias:0&decoder/layer_6/residual_0/bias/Assign5decoder/layer_6/residual_0/bias/Read/ReadVariableOp:0(23decoder/layer_6/residual_0/bias/Initializer/zeros:08
З
-decoder/layer_6/residual_1/depthwise_kernel:02decoder/layer_6/residual_1/depthwise_kernel/AssignAdecoder/layer_6/residual_1/depthwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_1/depthwise_kernel/Initializer/random_uniform:08
З
-decoder/layer_6/residual_1/pointwise_kernel:02decoder/layer_6/residual_1/pointwise_kernel/AssignAdecoder/layer_6/residual_1/pointwise_kernel/Read/ReadVariableOp:0(2Hdecoder/layer_6/residual_1/pointwise_kernel/Initializer/random_uniform:08
╗
!decoder/layer_6/residual_1/bias:0&decoder/layer_6/residual_1/bias/Assign5decoder/layer_6/residual_1/bias/Read/ReadVariableOp:0(23decoder/layer_6/residual_1/bias/Initializer/zeros:08
╩
%decoder/layer_6/ln/layer_norm_scale:0*decoder/layer_6/ln/layer_norm_scale/Assign9decoder/layer_6/ln/layer_norm_scale/Read/ReadVariableOp:0(26decoder/layer_6/ln/layer_norm_scale/Initializer/ones:08
К
$decoder/layer_6/ln/layer_norm_bias:0)decoder/layer_6/ln/layer_norm_bias/Assign8decoder/layer_6/ln/layer_norm_bias/Read/ReadVariableOp:0(26decoder/layer_6/ln/layer_norm_bias/Initializer/zeros:08
е
autoencoder_final/kernel:0autoencoder_final/kernel/Assign.autoencoder_final/kernel/Read/ReadVariableOp:0(25autoencoder_final/kernel/Initializer/random_uniform:08
Ќ
autoencoder_final/bias:0autoencoder_final/bias/Assign,autoencoder_final/bias/Read/ReadVariableOp:0(2*autoencoder_final/bias/Initializer/zeros:08"F
hub_module_attachments,*


stamp_size
ђ


pixel_size
Ј┬ш<*Ј
defaultЃ
7
default,
Placeholder:0         H
default=
autoencoder_final/Softplus:0         ђђ