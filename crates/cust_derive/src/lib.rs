//! Custom derive macro crate for cust.

#[macro_use]
extern crate quote;
extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;

use proc_macro2::{Ident, Span, TokenStream};
use syn::{
    Data, DataEnum, DataStruct, DataUnion, DeriveInput, Field, Fields, Generics, TypeParamBound,
    parse_str,
};

#[proc_macro_derive(DeviceCopyCore)]
pub fn device_copy_core(input: BaseTokenStream) -> BaseTokenStream {
    let ast = syn::parse(input).unwrap();
    let code = impl_device_copy(&ast, quote!(::cust_core::DeviceCopy));
    BaseTokenStream::from(code)
}
#[proc_macro_derive(DeviceCopy)]
pub fn device_copy(input: BaseTokenStream) -> BaseTokenStream {
    let ast = syn::parse(input).unwrap();
    let code = impl_device_copy(&ast, quote!(::cust::memory::DeviceCopy));
    BaseTokenStream::from(code)
}

use proc_macro::TokenStream as BaseTokenStream;

fn impl_device_copy(input: &DeriveInput, import: TokenStream) -> TokenStream {
    let input_type = &input.ident;

    // Generate the code to type-check all fields of the derived struct/enum/union. We can't perform
    // type checking at expansion-time, so instead we generate a dummy nested function with a
    // type-bound on DeviceCopy and call it with every type that's in the struct/enum/union.
    // This will fail to compile if any of the nested types doesn't implement DeviceCopy.
    let check_types_code = match input.data {
        Data::Struct(ref data_struct) => type_check_struct(data_struct),
        Data::Enum(ref data_enum) => type_check_enum(data_enum),
        Data::Union(ref data_union) => type_check_union(data_union),
    };

    // We need a function for the type-checking code to live in, so generate a complicated and
    // hopefully-unique name for that
    let type_test_func_name = format!(
        "__verify_{}_can_implement_devicecopy",
        input_type.to_string().to_lowercase()
    );
    let type_test_func_ident = Ident::new(&type_test_func_name, Span::call_site());

    // If the struct/enum/union is generic, we need to add the DeviceCopy bound to the generics
    // when implementing DeviceCopy.
    let generics = add_bound_to_generics(&input.generics, import.clone());
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    // Finally, generate the unsafe impl and the type-checking function.
    let generated_code = quote! {
        unsafe impl #impl_generics #import for #input_type #type_generics #where_clause {}

        #[doc(hidden)]
        #[allow(all)]
        fn #type_test_func_ident #impl_generics(value: &#input_type #type_generics) #where_clause {
            fn assert_impl<T: #import>() {}
            #check_types_code
        }
    };

    generated_code
}

fn add_bound_to_generics(generics: &Generics, import: TokenStream) -> Generics {
    let mut new_generics = generics.clone();
    let bound: TypeParamBound = parse_str(&quote! {#import}.to_string()).unwrap();

    for type_param in &mut new_generics.type_params_mut() {
        type_param.bounds.push(bound.clone())
    }

    new_generics
}

fn type_check_struct(s: &DataStruct) -> TokenStream {
    let checks = match s.fields {
        Fields::Named(ref named_fields) => {
            let fields: Vec<&Field> = named_fields.named.iter().collect();
            check_fields(&fields)
        }
        Fields::Unnamed(ref unnamed_fields) => {
            let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
            check_fields(&fields)
        }
        Fields::Unit => vec![],
    };
    quote!(
        #(#checks)*
    )
}

fn type_check_enum(s: &DataEnum) -> TokenStream {
    let mut checks = vec![];

    for variant in &s.variants {
        match variant.fields {
            Fields::Named(ref named_fields) => {
                let fields: Vec<&Field> = named_fields.named.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unnamed(ref unnamed_fields) => {
                let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unit => {}
        }
    }
    quote!(
        #(#checks)*
    )
}

fn type_check_union(s: &DataUnion) -> TokenStream {
    let fields: Vec<&Field> = s.fields.named.iter().collect();
    let checks = check_fields(&fields);
    quote!(
        #(#checks)*
    )
}

fn check_fields(fields: &[&Field]) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let field_type = &field.ty;
            quote! {assert_impl::<#field_type>();}
        })
        .collect()
}
