# Types

Types! who doesn't love types, especially those that cause libnvvm to randomly segfault or loop forever!
Anyways, types are an integral part of the codegen and everything revolves around them and you will see them everywhere.

`rustc_codegen_ssa` does not actually tell you what your type representation should be, it allows you to decide. For
example, `rust-gpu` represents it as a `SpirvType` enum, while both `cg_llvm` and our codegen represent it as 
opaque llvm types:

```rs
type Type = &'ll llvm::Type;
```

`llvm::Type` is an opaque type that comes from llvm-c. `'ll` is one of the main lifetimes you will see
throughout the whole codegen, it is used for anything that lasts as long as the current usage of llvm. 
LLVM gives you back pointers when you ask for a type or value, some time ago `cg_llvm` fully switched to using
references over pointers, and we follow in their footsteps. 

One important fact about types is that they are opaque, you cannot take a type and ask "is this X struct?",
this is like asking "which chickens were responsible for this omelette?". You can ask if its a number type,
a vector type, a void type, etc. 

The SSA codegen needs to ask the backend for types for everything it needs to codegen MIR. It does 
this using a trait called `BaseTypeMethods`:

```rs
pub trait BaseTypeMethods<'tcx>: Backend<'tcx> {
    fn type_i1(&self) -> Self::Type;
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;
    fn type_isize(&self) -> Self::Type;

    fn type_f32(&self) -> Self::Type;
    fn type_f64(&self) -> Self::Type;

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_struct(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn type_ptr_to_ext(&self, ty: Self::Type, address_space: AddressSpace) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;

    /// Returns the number of elements in `self` if it is a LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize;

    fn float_width(&self, ty: Self::Type) -> usize;

    /// Retrieves the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
}
```

Every codegen implements this some way or another, you can find our implementation in `ty.rs`. Our
implementation is pretty straightforward, LLVM has functions that we link to which get us the types we need:

```rs
impl<'ll, 'tcx> BaseTypeMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn type_i1(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt1TypeInContext(self.llcx) }
    }

    fn type_i8(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt8TypeInContext(self.llcx) }
    }

    fn type_i16(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt16TypeInContext(self.llcx) }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt32TypeInContext(self.llcx) }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt64TypeInContext(self.llcx) }
    }

    fn type_i128(&self) -> &'ll Type {
        unsafe { llvm::LLVMIntTypeInContext(self.llcx, 128) }
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe { llvm::LLVMFloatTypeInContext(self.llcx) }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe { llvm::LLVMDoubleTypeInContext(self.llcx) }
    }

    fn type_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, False) }
    }

// ...
```

There is also logic for handling ABI types, such as generating aggregate (struct) types 
