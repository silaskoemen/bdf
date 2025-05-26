; ModuleID = 'autocfg_d10cc7bb58f8e033_1.2b042e0d73ea3aeb-cgu.0'
source_filename = "autocfg_d10cc7bb58f8e033_1.2b042e0d73ea3aeb-cgu.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

@alloc_f93507f8ba4b5780b14b2c2584609be0 = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\F0?", align 8
@alloc_ef0a1f828f3393ef691f2705e817091c = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\00@", align 8

; core::f64::<impl f64>::total_cmp
; Function Attrs: inlinehint uwtable
define internal i8 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17ha4fda64edffc48e5E"(ptr align 8 %self, ptr align 8 %other) unnamed_addr #0 !dbg !16 {
start:
  %other.dbg.spill = alloca [8 x i8], align 8
  %self.dbg.spill = alloca [8 x i8], align 8
  %_6 = alloca [8 x i8], align 8
  %_3 = alloca [8 x i8], align 8
  store ptr %self, ptr %self.dbg.spill, align 8
    #dbg_declare(ptr %self.dbg.spill, !25, !DIExpression(), !28)
  store ptr %other, ptr %other.dbg.spill, align 8
    #dbg_declare(ptr %other.dbg.spill, !26, !DIExpression(), !29)
  %_5 = load double, ptr %self, align 8, !dbg !30
  %_4 = bitcast double %_5 to i64, !dbg !31
  store i64 %_4, ptr %_3, align 8, !dbg !30
  %_8 = load double, ptr %other, align 8, !dbg !37
  %_7 = bitcast double %_8 to i64, !dbg !38
  store i64 %_7, ptr %_6, align 8, !dbg !37
  %_13 = load i64, ptr %_3, align 8, !dbg !40
  %_12 = ashr i64 %_13, 63, !dbg !41
  %_10 = lshr i64 %_12, 1, !dbg !42
  %0 = load i64, ptr %_3, align 8, !dbg !43
  %1 = xor i64 %0, %_10, !dbg !43
  store i64 %1, ptr %_3, align 8, !dbg !43
  %_18 = load i64, ptr %_6, align 8, !dbg !44
  %_17 = ashr i64 %_18, 63, !dbg !45
  %_15 = lshr i64 %_17, 1, !dbg !46
  %2 = load i64, ptr %_6, align 8, !dbg !47
  %3 = xor i64 %2, %_15, !dbg !47
  store i64 %3, ptr %_6, align 8, !dbg !47
  %_19 = load i64, ptr %_3, align 8, !dbg !48
  %_20 = load i64, ptr %_6, align 8, !dbg !58
  %_0 = call i8 @llvm.scmp.i8.i64(i64 %_19, i64 %_20), !dbg !59
  ret i8 %_0, !dbg !60
}

; autocfg_d10cc7bb58f8e033_1::probe
; Function Attrs: uwtable
define void @_ZN26autocfg_d10cc7bb58f8e033_15probe17h7aacd250c13aa2aeE() unnamed_addr #1 !dbg !61 {
start:
; call core::f64::<impl f64>::total_cmp
  %_1 = call i8 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17ha4fda64edffc48e5E"(ptr align 8 @alloc_f93507f8ba4b5780b14b2c2584609be0, ptr align 8 @alloc_ef0a1f828f3393ef691f2705e817091c), !dbg !66
  ret void, !dbg !67
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.scmp.i8.i64(i64, i64) #2

attributes #0 = { inlinehint uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #1 = { uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{!"rustc version 1.87.0 (17067e9ac 2025-05-09)"}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.87.0 (17067e9ac 2025-05-09))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "autocfg_d10cc7bb58f8e033_1/@/autocfg_d10cc7bb58f8e033_1.2b042e0d73ea3aeb-cgu.0", directory: "/Users/silaskoemen/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/num-traits-0.2.19")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Ordering", scope: !9, file: !8, baseType: !11, size: 8, align: 8, flags: DIFlagEnumClass, elements: !12)
!8 = !DIFile(filename: "<unknown>", directory: "")
!9 = !DINamespace(name: "cmp", scope: !10)
!10 = !DINamespace(name: "core", scope: null)
!11 = !DIBasicType(name: "i8", size: 8, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DIEnumerator(name: "Less", value: -1)
!14 = !DIEnumerator(name: "Equal", value: 0)
!15 = !DIEnumerator(name: "Greater", value: 1)
!16 = distinct !DISubprogram(name: "total_cmp", linkageName: "_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17ha4fda64edffc48e5E", scope: !18, file: !17, line: 1350, type: !20, scopeLine: 1350, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !27, retainedNodes: !24)
!17 = !DIFile(filename: "/rustc/17067e9ac6d7ecb70e50f92c1944e545188d2359/library/core/src/num/f64.rs", directory: "", checksumkind: CSK_MD5, checksum: "73b19c8a0b69a817deddbd19d71bc223")
!18 = !DINamespace(name: "{impl#0}", scope: !19)
!19 = !DINamespace(name: "f64", scope: !10)
!20 = !DISubroutineType(types: !21)
!21 = !{!7, !22, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&f64", baseType: !23, size: 64, align: 64, dwarfAddressSpace: 0)
!23 = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
!24 = !{!25, !26}
!25 = !DILocalVariable(name: "self", arg: 1, scope: !16, file: !17, line: 1350, type: !22)
!26 = !DILocalVariable(name: "other", arg: 2, scope: !16, file: !17, line: 1350, type: !22)
!27 = !{}
!28 = !DILocation(line: 1350, column: 22, scope: !16)
!29 = !DILocation(line: 1350, column: 29, scope: !16)
!30 = !DILocation(line: 1351, column: 24, scope: !16)
!31 = !DILocation(line: 1095, column: 18, scope: !32, inlinedAt: !36)
!32 = distinct !DISubprogram(name: "to_bits", linkageName: "_ZN4core3f6421_$LT$impl$u20$f64$GT$7to_bits17he249a814aa51e6d7E", scope: !18, file: !17, line: 1093, type: !33, scopeLine: 1093, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !27)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !23}
!35 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!36 = !DILocation(line: 1351, column: 29, scope: !16)
!37 = !DILocation(line: 1352, column: 25, scope: !16)
!38 = !DILocation(line: 1095, column: 18, scope: !32, inlinedAt: !39)
!39 = !DILocation(line: 1352, column: 31, scope: !16)
!40 = !DILocation(line: 1376, column: 20, scope: !16)
!41 = !DILocation(line: 1376, column: 19, scope: !16)
!42 = !DILocation(line: 1376, column: 17, scope: !16)
!43 = !DILocation(line: 1376, column: 9, scope: !16)
!44 = !DILocation(line: 1377, column: 21, scope: !16)
!45 = !DILocation(line: 1377, column: 20, scope: !16)
!46 = !DILocation(line: 1377, column: 18, scope: !16)
!47 = !DILocation(line: 1377, column: 9, scope: !16)
!48 = !DILocation(line: 1939, column: 58, scope: !49, inlinedAt: !57)
!49 = distinct !DISubprogram(name: "cmp", linkageName: "_ZN4core3cmp5impls48_$LT$impl$u20$core..cmp..Ord$u20$for$u20$i64$GT$3cmp17h235dd35de641c986E", scope: !51, file: !50, line: 1938, type: !53, scopeLine: 1938, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !27)
!50 = !DIFile(filename: "/rustc/17067e9ac6d7ecb70e50f92c1944e545188d2359/library/core/src/cmp.rs", directory: "", checksumkind: CSK_MD5, checksum: "0baf28632b40126315e43d523424d42e")
!51 = !DINamespace(name: "{impl#79}", scope: !52)
!52 = !DINamespace(name: "impls", scope: !9)
!53 = !DISubroutineType(types: !54)
!54 = !{!7, !55, !55}
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&i64", baseType: !56, size: 64, align: 64, dwarfAddressSpace: 0)
!56 = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
!57 = !DILocation(line: 1379, column: 14, scope: !16)
!58 = !DILocation(line: 1939, column: 65, scope: !49, inlinedAt: !57)
!59 = !DILocation(line: 1939, column: 21, scope: !49, inlinedAt: !57)
!60 = !DILocation(line: 1380, column: 6, scope: !16)
!61 = distinct !DISubprogram(name: "probe", linkageName: "_ZN26autocfg_d10cc7bb58f8e033_15probe17h7aacd250c13aa2aeE", scope: !63, file: !62, line: 1, type: !64, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, templateParams: !27)
!62 = !DIFile(filename: "<anon>", directory: "", checksumkind: CSK_MD5, checksum: "ca821b87a81998bc0a84ab6029e9650c")
!63 = !DINamespace(name: "autocfg_d10cc7bb58f8e033_1", scope: null)
!64 = !DISubroutineType(types: !65)
!65 = !{null}
!66 = !DILocation(line: 1, column: 26, scope: !61)
!67 = !DILocation(line: 1, column: 50, scope: !68)
!68 = !DILexicalBlockFile(scope: !61, file: !62, discriminator: 0)
