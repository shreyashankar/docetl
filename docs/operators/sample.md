# Sample operation

The Sample operation in DocETL samples items from the input. It is
meant mostly as a debugging tool:

Insert it before the last operation, the one you're currently trying
to tack on to the end of a working pipeline, to limit the amount of
data it will be fed, so that the run time is small enough to
comfortably debug its prompt. Once it seems to be working, you can
remove the sample operation. You can then repeat this for each
operation you add while developing your pipeline!



## 🚀 Example: 

```yaml
- name: cluster_concepts
  type: sample
  samples: 0.1
  random_state: 42
  stratify: category
```

This sample operation will return a pseudo-randomly selected 10% of
the samples (`samples: 0.1`). The random selection will be seeded with
a constant (42), meaning the same selection will be returned if you
rerun the pipeline (If no random state is given, a different sample
will be returned every time). Additionally, the random sampling will
sample each value of the `category` key equally.

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "sample".
- `samples`: Either a list of sample indices to just return those samples, an integer count of samples, or a float fraction of samples.

## Optional Parameters

| Parameter                 | Description                                                                      | Default                       |
| ------------------------- | -------------------------------------------------------------------------------- | ----------------------------- |
| `random_state             | An integer to seed the random generator with | Use the (numpy) global random state
| `stratify`                | The key to stratify by             |  |
