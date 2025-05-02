weights={
"GlutWT":{
"peakweight":      1000000,
"clweight":        1000000000,
"phweight":        1000000000000000,
"transportweight": 100,
"pkaweight":       1000,
},
"AspWT":{
"peakweight":      1000000,
"clweight":        100000,
"phweight":        100000,
"transportweight": 100,
"pkaweight":       200,
},
 # time course data exists in up to 3 segments, only existing & relevant ones have nonzero weight
"WTintGlut40Cl_pH55":      {0:1000000, 1:0, 2:0},
"WTintGlut40Cl_pH5":       {0:1000000, 1:0, 2:0},
"WTintGlut40Cl_pH5App":    {0:1000, 1:2000000000, 2:2000000000},
"WTintGlutpH5_40ClApp":    {0:10000, 1:3000000, 2:3000000},
"WTintGlutpH55_140ClApp":  {0:5000, 1:5000, 2:0},
"WTintGlutpH55_140ClApp2": {0:0, 1:5000, 2:0},
"WTintAsp40Cl_pH5":        {0:1000, 1:0, 2:0},
"WTintAsp40Cl_pH5App":     {0:20000, 1:80000000000000, 2:1000000000},
"WTintAsppH55_40ClApp":    {0:1000, 1:800000000, 2:800000000},
}
