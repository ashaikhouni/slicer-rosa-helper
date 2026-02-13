# Publish Checklist

## 1. Initialize branch and first commit

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
git branch -m main
git add .
git commit -m "Initial ROSA Helper module"
```

## 2. Create GitHub repository

Create repository `slicer-rosa-helper` on GitHub, then:

```bash
git remote add origin git@github.com:<org-or-user>/slicer-rosa-helper.git
git push -u origin main
```

## 3. Optional release tag

```bash
git tag -a v0.1.0 -m "ROSA Helper v0.1.0"
git push origin v0.1.0
```

## 4. User install instructions

Share `INSTALL.md` and tell users to add this path in Slicer:

`<repo>/RosaHelper`

## 5. Optional extension catalog publication (later)

When ready for Extension Manager distribution:
- add/verify extension metadata
- submit extension descriptor PR to `Slicer/ExtensionsIndex`
