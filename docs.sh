# Env
source ~/piqa/bin/activate

# Master
git checkout master

# Deactivate JIT
sed -i 's/torch.jit.script/lambda f: f/g' piqa/utils.py

# Generate HTML
pdoc piqa --html --force \
	--config "git_link_template='https://github.com/francois-rozet/piqa/blob/{commit}/{path}#L{start_line}-L{end_line}'" \
	--config "show_inherited_members=True" \
	--config "latex_math=True"

# Docs
git restore piqa/utils.py
git checkout docs
git reset --hard master

mkdir docs -p
mv html/piqa/* docs/
rm -rf html

git add docs/*
git commit -m "📝 HTLM documentation"
