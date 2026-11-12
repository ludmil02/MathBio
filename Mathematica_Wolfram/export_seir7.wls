(* Evaluate the setup cells of the SEIR 7 notebook needed for the next generation
   matrix and export the data to CSV.  Usage: wolframscript -file export_seir7.wls NB OUTDIR T *)
x`nbfile = $ScriptCommandLine[[2]]; x`out = $ScriptCommandLine[[3]]; x`T = ToExpression[$ScriptCommandLine[[4]]];
x`nb = Get[x`nbfile];
x`cells = Cases[x`nb, Cell[c_, st:("Input"|"Code"|"Section"|"Subsection"|"Subsubsection"|"Title"|"Text"), ___] :> {st, c}, Infinity];
x`hold[BoxData[x`b_List]] := Join @@ (ToExpression[#, StandardForm, HoldComplete] & /@ x`b);
x`hold[BoxData[x`b_]] := ToExpression[x`b, StandardForm, HoldComplete];
run[k_] := (Print[DateString[{"Time"}], "  cell ", k]; 
   Block[{SetDirectory = (Null &)}, Scan[ReleaseHold, List @@ (HoldComplete /@ Unevaluated @@ {hold[cells[[k, 2]]]})]]);
x`evalCell[x`k_] := Module[{x`h = x`hold[x`cells[[x`k, 2]]]},
   Print[DateString[{"Time"}], "  cell ", x`k, " (", Length[x`h], " statements)"];
   Block[{SetDirectory = (Null &)}, Do[ReleaseHold[Extract[x`h, x`e, HoldComplete]], {x`e, Length[x`h]}]]];
Scan[x`evalCell, {2, 5, 6, 7, 8, 11, 13, 14, 16, 18, 20, 21, 30, 31, 32, 44, 45, 46, 47}];
Print["GeoDistance unit: ", QuantityUnit[GeoDistance[Locations[[1]], Locations[[2]]]]];
Print["b = ", b];
(* cell 53 without the ParallelTable for mc; mc is formed in Python, slice i = 1 checked *)
DM = Table[DiagonalMatrix[ConstantArray[LR[[p]], n]], {p, 1, m}];
mC = Table[DM[[p]] . USC, {p, 1, m}];
ma = Table[Sum[mC[[p, j, q]], {j, 1, n}], {p, 1, m}, {q, 1, n}];
mc1 = Table[(mC[[p, j, q]]*Exp[-b[[q]]*dist[[i, p]]])/Sum[mC[[s, j, r]]*Exp[-b[[r]]*dist[[i, s]]], {r, 1, n}, {s, 1, m}], {i, {1, 33}}, {j, 1, n}, {p, 1, m}, {q, 1, n}];
Print[DateString[{"Time"}], "  mc slices i = 1, 33 done"];
Scan[x`evalCell, {56, 57, 62}];
vSv = Table[pred[i, x`T], {i, 1, m}];
Print["m = ", m, ", n = ", n, ", T = ", x`T];
Print["vS[x`T] range: ", MinMax[vSv]];
ex[name_, y_] := (Export[FileNameJoin[{x`out, name <> ".csv"}], y, "CSV"]; Print["  wrote ", name, " ", Dimensions[y]]);
ex["mN", mN]; ex["USC", USC]; ex["LR", List /@ LR]; ex["dist", dist]; ex["b", List /@ b];
ex["ma", ma]; ex["mc_slices", Flatten[mc1, {{1, 2}, {3, 4}}]];
ex["vS", List /@ vSv];
ex["vpsiS", List /@ v\[Psi]S[x`T]]; ex["vpsiI", List /@ v\[Psi]I[x`T]]; ex["vbeta", List /@ v\[Beta][x`T]];
ex["vp", List /@ vp]; ex["vq", List /@ vq];
ex["vgamma", Transpose[{v\[Gamma]a, v\[Gamma]s, v\[Gamma]m, v\[Gamma]h}]]; ex["vdelta", Transpose[{v\[Delta]s, v\[Delta]m}]];
ex["scalars", {{"k", k}, {"xi", \[Xi]}, {"omega", \[Omega]}, {"omega_v", \[Omega]v}, {"eps1", \[Epsilon]1[x`T]}, {"eps2", \[Epsilon]2[x`T]}, {"eps3", \[Epsilon]3[x`T]},
   {"eta_a", \[Eta]a}, {"eta_ps", \[Eta]ps}, {"eta_pm", \[Eta]pm}, {"eta_s", \[Eta]s}, {"eta_m", \[Eta]m}, {"eta_h", \[Eta]h}}];
Print[DateString[{"Time"}], "  done"];
