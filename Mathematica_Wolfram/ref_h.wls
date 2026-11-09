(* evaluate the input cells of eigenvalues_H_ZF_JG.nb per chi; Export and SetDirectory disabled *)
c`nbfile = $ScriptCommandLine[[2]]; c`fin = $ScriptCommandLine[[3]]; c`fout = $ScriptCommandLine[[4]];
SetDirectory[DirectoryName[c`fout]];
c`nb = Get[c`nbfile];
(* the notebook's first input cell loads settings_zf_jg.wl; here it is read once, held, and run per chi *)
c`settings = Import[FileNameJoin[{DirectoryName[c`nbfile], "settings_zf_jg.wl"}], "HeldExpressions"];
c`ins = Cases[c`nb, Cell[BoxData[c`b_], "Input", ___] :> c`b, Infinity];
c`run[c`x1_, c`x2_, c`x3_] := Module[{c`h},
  ClearAll["Global`*"];
  Quiet@Block[{Export = (Null &), SetDirectory = (Null &)},
    Scan[ReleaseHold[# /. {HoldPattern[Set[Global`m\[Chi]1, _]] :> Set[Global`m\[Chi]1, c`x1],
                           HoldPattern[Set[Global`m\[Chi]2, _]] :> Set[Global`m\[Chi]2, c`x2],
                           HoldPattern[Set[Global`m\[Chi]3, _]] :> Set[Global`m\[Chi]3, c`x3]}] &, c`settings];
    Scan[Function[c`b,
      c`h = ToExpression[c`b, StandardForm, HoldComplete];
      c`h = c`h /. {HoldPattern[Set[Global`m\[Chi]1, _]] :> Set[Global`m\[Chi]1, c`x1],
                    HoldPattern[Set[Global`m\[Chi]2, _]] :> Set[Global`m\[Chi]2, c`x2],
                    HoldPattern[Set[Global`m\[Chi]3, _]] :> Set[Global`m\[Chi]3, c`x3]};
      ReleaseHold[c`h]], c`ins]];
  <|"NGM" -> N[Global`NGM], "Y" -> N[Global`Y], "Z" -> N[Global`Z],
    "rho1" -> Re[First[Eigenvalues[Global`NGM, 1]]],
    "ev" -> ({Re[#], Im[#]} & /@ Eigenvalues[SparseArray[N[Global`NGM]], 16, Method -> {"Arnoldi", "MaxIterations" -> 100000}])|>];
c`cases = Import[c`fin, "RawJSON"];
c`out = Table[Join[<|"name" -> c`cs["name"]|>, c`run[N[c`cs["chi1"]], N[c`cs["chi2"]], N[c`cs["chi3"]]]], {c`cs, c`cases}];
Export[c`fout, c`out, "JSON", "Compact" -> True];
Print["cases evaluated in the H notebook: ", Length[c`out]];
