#!/bin/bash

# 提示用户输入提交消息
echo "Enter commit message:"
read commit_message

# 执行 git add, git commit 和 git push 操作
git add .
git commit -m "$commit_message"
git push origin main  # 或者你可以修改为你的分支名称
