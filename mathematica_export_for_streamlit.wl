(* ::Package:: *)
(* Heartfield Mathematica -> Streamlit CSV export helper *)

ClearAll[toRowList, assocRowsQ, exportRows];

toRowList[data_] := Which[
  Head[data] === Dataset, Normal[data],
  AssociationQ[data], Values[data],
  ListQ[data], data,
  True, {}
];

assocRowsQ[rows_] := ListQ[rows] && (Length[rows] == 0 || AllTrue[rows, AssociationQ]);

exportRows[file_String, rows_List] := Module[{path = FileNameJoin[{Directory[], "data", file}]},
  If[!assocRowsQ[rows], Return[$Failed]];
  If[Length[rows] == 0, Return[$Failed]];
  Export[path, rows, "CSV"];
  path
];

(* 1) Predictive accuracy (preferred variables from notebook) *)
If[NameQ["Global`l2ShuffleReplicates"],
  Module[{rows, shuffledAcc, histAcc, predRows},
    rows = toRowList[Evaluate[Global`l2ShuffleReplicates]];
    If[assocRowsQ[rows] && Length[rows] > 0,
      shuffledAcc = N @ Mean[Lookup[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Shuffled" &], "Accuracy"]];
      histAcc = N @ Mean[Lookup[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Hist" &], "Accuracy"]];
      predRows = {
        <|"feature_level" -> "L2_Shuffled", "mean_accuracy" -> shuffledAcc, "std" -> N @ StandardDeviation[Lookup[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Shuffled" &], "Accuracy"]], "n_reps" -> Length[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Shuffled" &]], "is_control" -> True|>,
        <|"feature_level" -> "L2_Hist", "mean_accuracy" -> histAcc, "std" -> N @ StandardDeviation[Lookup[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Hist" &], "Accuracy"]], "n_reps" -> Length[Select[rows, Lookup[#, "FeatureLevel", ""] === "L2_Hist" &]], "is_control" -> False|>
      };
      exportRows["predictive_accuracy.csv", predRows];
    ];
  ];
];

(* 2) Lifecycle from ablation rows, if present *)
If[NameQ["Global`ablationRows"],
  Module[{rows, grouped, out},
    rows = toRowList[Evaluate[Global`ablationRows]];
    If[assocRowsQ[rows] && Length[rows] > 0,
      grouped = GroupBy[
        rows,
        {Lookup[#, "FeatureLevel", "Unknown"] &, Lookup[#, "Layer", Missing["NA"]] &}
      ];
      out = KeyValueMap[
        Function[{k, vals},
          <|
            "level" -> k[[1]],
            "generation" -> k[[2]],
            "mean_radius" -> N @ Mean[Lookup[vals, "MeanRadius"]],
            "state_count" -> N @ Mean[Lookup[vals, "DistinctTuples", Lookup[vals, "LayerStateCount", 0]]]
          |>
        ],
        grouped
      ];
      exportRows["lifecycle_by_level.csv", out];
    ];
  ];
];

(* 3) Coarse-rich divergence from agency rows, if present *)
If[NameQ["Global`agencyRows3"],
  Module[{rows, out},
    rows = toRowList[Evaluate[Global`agencyRows3]];
    If[assocRowsQ[rows] && Length[rows] > 0,
      out = Map[
        <|
          "layer" -> Lookup[#, "Layer", Missing["NA"]],
          "distinct_expr" -> Lookup[#, "TerminalUnionExprCount", Missing["NA"]],
          "distinct_coarse" -> Lookup[#, "TerminalUnionCoarseTupleCount", Missing["NA"]],
          "distinct_rich" -> Lookup[#, "TerminalUnionRichTupleCount", Missing["NA"]],
          "coarse_entropy_norm" -> Lookup[#, "NormalizedCoarseEntropy", Missing["NA"]],
          "rich_entropy_norm" -> Lookup[#, "NormalizedRichEntropy", Missing["NA"]]
        |> &,
        rows
      ];
      exportRows["coarse_rich_divergence.csv", out];
    ];
  ];
];

(* 4) 3D embedding from rich event rows, if present *)
If[NameQ["Global`richEventRows"],
  Module[{rows, out},
    rows = toRowList[Evaluate[Global`richEventRows]];
    If[assocRowsQ[rows] && Length[rows] > 0,
      out = Select[
        Map[
          <|
            "x" -> Lookup[#, "RPC1", Missing["NA"]],
            "y" -> Lookup[#, "RPC2", Missing["NA"]],
            "z" -> Lookup[#, "RPC3", Missing["NA"]],
            "operator" -> Lookup[#, "Operator", "Unknown"]
          |> &,
          rows
        ],
        NumericQ[Lookup[#, "x"]] && NumericQ[Lookup[#, "y"]] && NumericQ[Lookup[#, "z"]] &
      ];
      exportRows["embedding_points.csv", out];
    ];
  ];
];

Print["Export pass complete. Check ./data for CSVs used by Streamlit."];
