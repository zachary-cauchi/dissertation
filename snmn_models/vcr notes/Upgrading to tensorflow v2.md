Create a copy of all python scripts to convert:

```bash
find snmn_models -name '*.py' -exec cp --parents \{\} snmn_models-v1 \;
```

Execute preliminary upgrade script:
```bash
tf_upgrade_v2 --intree snmn_models-v1/ --outtree snmn_models-v2 --reportfile report.txt
```
