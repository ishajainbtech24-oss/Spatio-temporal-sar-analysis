# data/

Not tracked by git. Expected layout after downloading Sen1Floods11 (see `docs/DATASET.md`):

```
data/raw/sen1floods11/v1.1/
├── data/flood_events/HandLabeled/S1Hand/     *_S1Hand.tif
├── data/flood_events/HandLabeled/LabelHand/  *_LabelHand.tif
└── splits/flood_handlabeled/                 flood_{train,valid,test}_data.csv
```

`sar-flood make-synthetic` writes a small synthetic dataset to `data/synthetic/` for smoke tests.
