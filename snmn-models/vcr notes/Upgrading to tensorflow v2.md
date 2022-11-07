Create a copy of all python scripts to convert:

```bash
find snmn-models -name '*.py' -exec cp --parents \{\} snmn-models-v1 \;
```

Execute preliminary upgrade script:
```bash
tf_upgrade_v2 --intree snmn-models-v1/ --outtree snmn-models-v2 --reportfile report.txt
```
