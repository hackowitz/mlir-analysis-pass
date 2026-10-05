target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define i32 @sqlite3VdbeMemTranslate(i32 %0) {
  %2 = and i32 %0, 3
  %3 = add i32 216, %2
  ret i32 %3
}
