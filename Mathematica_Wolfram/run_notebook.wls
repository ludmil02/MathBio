(* Evaluates the input cells of a notebook in order, without a front end, and
   prints every result that is not Null, together with whatever the cells Print.
   The notebook is not modified; NotebookDirectory[] is the notebook's directory,
   and Export is disabled, so nothing is written.  Graphics are reported by head.

   Usage:
       wolframscript -file run_notebook.wls NOTEBOOK.nb > NOTEBOOK.log           *)

c`nbfile = AbsoluteFileName[$ScriptCommandLine[[2]]];
c`nb = Get[c`nbfile];
(* input cells come as boxes, Cell[BoxData[...], "Input"], or as strings, Cell["...", "Input"] *)
c`ins = Cases[c`nb, Cell[c`c_, "Input", ___] :> c`c, Infinity];
c`hold[BoxData[c`b_List]] := Join @@ (ToExpression[#, StandardForm, HoldComplete] & /@ c`b);
c`hold[BoxData[c`b_]] := ToExpression[c`b, StandardForm, HoldComplete];
c`hold[c`s_String] := ToExpression[c`s, InputForm, HoldComplete];
Print["notebook ", c`nbfile, ", ", Length[c`ins], " input cells"];
Block[{NotebookDirectory = (DirectoryName[c`nbfile] &), Export = (Null &)},
  Do[
    (* a cell may hold several statements: evaluate them in order, print each non-Null result *)
    c`h = c`hold[c`ins[[c`k]]];
    Do[
      c`r = ReleaseHold[Extract[c`h, c`e, HoldComplete]];
      Scan[If[# =!= Null,
          Print["cell ", c`k, ": ",
            Which[
              MatchQ[#, _Graphics | _Graphics3D | _Legended | _GraphicsGrid | _GraphicsRow], Head[#],
              MatchQ[#, _Grid], ToString[TableForm[First[#]], OutputForm],
              True, ToString[#, OutputForm]]]] &,
        If[ListQ[c`r] && MemberQ[c`r, Null], c`r, {c`r}]],
      {c`e, Length[c`h]}],
    {c`k, Length[c`ins]}]];
