use proc_macro::TokenStream;
use quote::{ToTokens, quote_spanned};
use syn::{
    FnArg, Ident, ItemFn, ReturnType, Stmt, parse_macro_input, parse_quote, spanned::Spanned,
};

/// Registers a function as a gpu kernel.
///
/// This attribute must always be placed on gpu kernel functions.
///
/// This attribute does a couple of things:
/// - Tells `rustc_codegen_nvvm` to mark this as a gpu kernel and to not remove it from the ptx
///   file.
/// - Marks the function as `no_mangle`.
/// - Errors if the function is not unsafe.
/// - Makes sure function parameters are all [`Copy`].
/// - Makes sure the function doesn't return anything.
///
/// Note that this does not cfg the function for nvptx(64), that is explicit so that rust analyzer
/// is able to offer intellisense by default.
#[proc_macro_attribute]
pub fn kernel(input: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    let cloned = input.clone();
    let input = parse_macro_input!(cloned as proc_macro2::TokenStream);
    let mut item = parse_macro_input!(item as ItemFn);
    let no_mangle = parse_quote!(#[unsafe(no_mangle)]);
    item.attrs.push(no_mangle);
    let internal = parse_quote!(#[cfg_attr(target_arch="nvptx64", nvvm_internal::kernel(#input))]);
    item.attrs.push(internal);

    // Used to guarantee some things about how params are passed in the codegen.
    item.sig.abi = Some(parse_quote!(extern "C"));

    let check_fn = parse_quote! {
        fn assert_kernel_parameter_is_copy<T: Copy>() {}
    };
    item.block.stmts.insert(0, check_fn);

    for param in &item.sig.inputs {
        let ty = match param {
            FnArg::Receiver(_) => quote_spanned! {
                param.span() => ::core::compile_error!("Kernel functions may not be struct methods");
            },
            FnArg::Typed(ty) => ty.ty.to_token_stream(),
        };
        let call = parse_quote! {
            assert_kernel_parameter_is_copy::<#ty>();
        };
        item.block.stmts.insert(0, call);
    }

    let ret = item.sig.output.clone();
    if let ReturnType::Type(_, _) = ret {
        let err = quote_spanned! {
            ret.span() => ::core::compile_err!("Kernel functions should not return anything");
        }
        .into();
        item.block.stmts.insert(0, parse_macro_input!(err as Stmt));
    }

    if item.sig.unsafety.is_none() {
        let err = quote_spanned! {
            item.span() => ::core::compile_error!("Kernel functions must be marked as unsafe");
        }
        .into();
        item.block.stmts.insert(0, parse_macro_input!(err as Stmt));
    }

    item.to_token_stream().into()
}

// derived from rust-gpu's gpu_only

/// Creates a cpu version of the function which panics and cfg-gates the function for only
/// nvptx/nvptx64.
#[proc_macro_attribute]
pub fn gpu_only(_attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    let syn::ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = syn::parse_macro_input!(item as syn::ItemFn);

    let mut cloned_attrs = attrs.clone();
    cloned_attrs.retain(|a| a.path().segments[0].ident != "nvvm_internal");

    let fn_name = sig.ident.clone();

    let sig_cpu = syn::Signature {
        abi: None,
        ..sig.clone()
    };

    let output = quote::quote! {
        #[cfg(not(target_arch="nvptx64"))]
        #[allow(unused_variables)]
        #(#cloned_attrs)* #vis #sig_cpu {
            unimplemented!(concat!("`", stringify!(#fn_name), "` can only be used on the GPU with rustc_codegen_nvvm"))
        }

        #[cfg(target_arch="nvptx64")]
        #(#attrs)* #vis #sig {
            #block
        }
    };

    output.into()
}

/// Notifies the codegen that this function is externally visible and should not be
/// removed if it is not used by a kernel. Usually used for linking with other PTX/cubin files.
///
/// # Panics
///
/// Panics if the function is not also no_mangle.
#[proc_macro_attribute]
pub fn externally_visible(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> TokenStream {
    let mut func = syn::parse_macro_input!(item as syn::ItemFn);

    assert!(
        func.attrs.iter().any(|a| a.path().is_ident("no_mangle")),
        "#[externally_visible] function should also be #[no_mangle]"
    );

    let new_attr = parse_quote!(#[cfg_attr(target_os = "cuda", nvvm_internal::used)]);
    func.attrs.push(new_attr);

    func.into_token_stream().into()
}

/// Notifies the codegen to put a `static`/`static mut` inside of a specific memory address space.
/// This is mostly for internal use and/or advanced users, as the codegen and `cuda_std` handle
/// address space placement implicitly. **Improper use of this macro could yield weird or undefined
/// behavior**.
///
/// This macro takes a single argument which can either be `global`, `shared`, `constant`, or
/// `local`.
///
/// This macro does nothing on the CPU.
///
/// # Shared memory
///
/// The item `#[address_space(shared) static mut FOO: [MaybeUninit<T>; N];` statically allocates a
/// buffer large enough for `N` elements of type `T`, yielding an uninitialized array in shared
/// memory.
///
/// Note that this allocates the memory __statically__, i.e. it expands to a static in the `shared`
/// address space. Therefore, calling this macro multiple times in a loop will always yield the
/// same data. However, separate invocations of the macro will yield different buffers.
///
/// Because the data is uninitialized by default, the type within the array must be `MaybeUninit`,
/// and uses must follow the usual rules of `MaybeUninit`, such as using `write`/`assume_init`.
/// Using a non-`MaybeUninit` type is undefined behaviour.
///
/// # Safety
///
/// Shared memory usage is fundamentally unsafe and much of the burden of correctness is on the
/// user. For example:
/// - Shared memory is only shared across __thread blocks__, not the entire device, therefore it is
///   unsound to rely on sharing data across more than one block.
/// - You must write to the shared buffer before reading from it as the data is uninitialized by
///   default.
/// - `cuda_std::thread::sync_threads` must be called before relying on the results of other
///   threads. This ensures every thread has reached that point before going on. For example, when
///   reading another thread's data after writing to the buffer.
///
/// It is suggested to run your executable in `cuda-memcheck` to make sure usages of
/// shared memory are right.
///
/// # Examples
///
/// ```ignore
/// use core::mem::MaybeUninit;
/// use cuda_std::*;
///
/// ##[kernel]
/// pub unsafe fn reverse_array(d: *mut u32, n: usize) {
///     ##[address_space(shared)]
///     static mut S: [MaybeUninit<u32>; 64] = [const { MaybeUninit::uninit() }; 64];
///     let i = thread::thread_idx_x() as usize;
///     let ir = n - i - 1;
///     unsafe { S[i].write(*d.add(i)); };
///     thread::sync_threads();
///     unsafe { *d.add(i) = S[ir].assume_init(); }
/// }
/// ```
#[proc_macro_attribute]
pub fn address_space(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> TokenStream {
    let mut global = syn::parse_macro_input!(item as syn::ItemStatic);
    let input = syn::parse_macro_input!(attr as Ident);

    let addrspace_num = match input.to_string().as_str() {
        "global" => 1,
        // what did you do to address space 2 libnvvm??
        "shared" => 3,
        "constant" => 4,
        "local" => 5,
        addr => panic!("Invalid address space `{}`", addr),
    };

    let new_attr =
        parse_quote!(#[cfg_attr(target_os = "cuda", nvvm_internal::addrspace(#addrspace_num))]);
    global.attrs.push(new_attr);

    global.into_token_stream().into()
}
