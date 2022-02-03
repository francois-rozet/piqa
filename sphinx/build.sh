#!/usr/bin/bash

# Exit at error
set -e

# Environment
source ~/piqa/bin/activate

# Merge
git checkout docs
git merge master -m "🔀 Merge master into docs"

# Generate HTML
sphinx-build -b html . ../docs
rm piqa*.rst

# Disable Jekyll
cd ../docs
touch .nojekyll

# Edit HTML
sed "s|<span class=\"pre\">\[source\]</span>|<i class=\"fa-solid fa-code\"></i>|g" -i *.html
sed "s|\(<a class=\"reference external\"[^¶]*</a>\)\(<a class=\"headerlink\".*¶</a>\)|\2\1|g" -i *.html
sed "s|<a class=\"muted-link\" href=\"https://pradyunsg.me\">@pradyunsg</a>'s||g" -i *.html
sed "s|Furo theme|Furo|g" -i *.html

# Push
git add .
git commit -m "📝 Update Sphinx documentation"
