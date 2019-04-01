
## Usage

### Model Comparison

```
$ ./model.sh | xargs -I{} bash -c "{}"
$ python ./vis_model_and_task.py -i ./result/all/[commit hash]
```

## Parameter Comparison

```
$ ./param.sh -d [dataset] | xargs -I{} bash -c "{}"
$ python ./vis_frac_and_duration.py -i ./result/bank/[commit hash]
```

