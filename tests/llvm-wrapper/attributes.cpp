#include "rustllvm.h"
#include "llvm/IR/Verifier.h"
#include <cstdlib>
#include "llvm/AsmParser/Parser.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

extern "C" void LLVMRustSetNormalizedTarget(LLVMModuleRef, const char *);
extern "C" void LLVMRustAddFunctionAttribute(LLVMValueRef, unsigned, LLVMRustAttribute);
extern "C" void LLVMRustAddFunctionAttributeWithType(LLVMValueRef, unsigned, LLVMRustAttribute, LLVMTypeRef);
extern "C" void LLVMRustRemoveFunctionAttributes(LLVMValueRef, unsigned, LLVMRustAttribute);
extern "C" void LLVMRustAddCallSiteAttribute(LLVMValueRef, unsigned, LLVMRustAttribute);

// These tests never request Rust-owned string output.
extern "C" void LLVMRustStringWriteImpl(RustStringRef, const char *, size_t) {
  std::abort();
}

static void require(bool Condition) {
  if (!Condition)
    std::abort();
}

extern "C" void LLVMRustRestoreNvvmKernelAnnotations(LLVMModuleRef);

static void kernelRetention() {
  llvm::LLVMContext C;
  llvm::SMDiagnostic Error;
  auto M = llvm::parseAssemblyString(R"(
    target triple = "nvptx64-nvidia-cuda"
    define void @entry(ptr %out) { store i32 42, ptr %out ret void }
    define internal void @dead() { ret void }
    !nvvm.annotations = !{!0}
    !0 = !{ptr @entry, !"kernel", i32 1}
  )", Error, C);
  require(bool(M));
  auto *F = M->getFunction("entry");
  require(F->getCallingConv() == llvm::CallingConv::PTX_Kernel);
  require(M->getNamedMetadata("nvvm.annotations")->getNumOperands() == 0);
  LLVMRustRestoreNvvmKernelAnnotations(llvm::wrap(M.get()));
  LLVMRustRestoreNvvmKernelAnnotations(llvm::wrap(M.get()));
  auto *Annotations = M->getNamedMetadata("nvvm.annotations");
  require(Annotations->getNumOperands() == 1);
  auto *Node = Annotations->getOperand(0);
  require(llvm::cast<llvm::ValueAsMetadata>(Node->getOperand(0))->getValue() == F);
  require(llvm::cast<llvm::MDString>(Node->getOperand(1))->getString() == "kernel");
  require(llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(2))->equalsInt(1));
  llvm::ModuleAnalysisManager AM;
  llvm::GlobalDCEPass().run(*M, AM);
  require(M->getFunction("entry") == F);
  require(M->getFunction("dead") == nullptr);
  require(!llvm::verifyModule(*M, &llvm::errs()));
}

int main() {
  kernelRetention();
  llvm::LLVMContext C;
  llvm::Module M("wrapper-test", C);
  LLVMRustSetNormalizedTarget(llvm::wrap(&M), "nvptx64-nvidia-cuda");
  require(M.getTargetTriple().str() == "nvptx64-nvidia-cuda");
  auto *Ptr = llvm::PointerType::get(C, 0);
  auto *FT = llvm::FunctionType::get(llvm::Type::getVoidTy(C), {Ptr}, false);
  auto *F = llvm::Function::Create(FT, llvm::GlobalValue::ExternalLinkage, "callee", M);
  auto CheckCapture = [](llvm::Attribute A) {
    require(A.isValid());
    require(A.getCaptureInfo() == llvm::CaptureInfo::none());
    require(A.getAsString() == "captures(none)");
  };
  LLVMRustAddFunctionAttribute(llvm::wrap(F), 1, NoCapture);
  CheckCapture(F->getAttributes().getParamAttr(0, llvm::Attribute::Captures));
  LLVMRustRemoveFunctionAttributes(llvm::wrap(F), 1, NoCapture);
  require(!F->hasParamAttribute(0, llvm::Attribute::Captures));
  LLVMRustAddFunctionAttributeWithType(llvm::wrap(F), 1, NoCapture, llvm::wrap(Ptr));
  CheckCapture(F->getAttributes().getParamAttr(0, llvm::Attribute::Captures));
  auto *Caller = llvm::Function::Create(FT, llvm::GlobalValue::ExternalLinkage, "caller", M);
  llvm::IRBuilder<> B(llvm::BasicBlock::Create(C, "entry", Caller));
  auto *Call = B.CreateCall(F, {Caller->getArg(0)});
  LLVMRustAddCallSiteAttribute(llvm::wrap(Call), 1, NoCapture);
  CheckCapture(Call->getAttributes().getParamAttr(0, llvm::Attribute::Captures));
  B.CreateRetVoid();
  require(!llvm::verifyModule(M, &llvm::errs()));
}
