# Kiro CLI pre block. Keep at the top of this file.
[[ -f "${HOME}/Library/Application Support/kiro-cli/shell/zshrc.pre.zsh" ]] && builtin source "${HOME}/Library/Application Support/kiro-cli/shell/zshrc.pre.zsh"
___MY_VMOPTIONS_SHELL_FILE="${HOME}/.jetbrains.vmoptions.sh"; if [ -f "${___MY_VMOPTIONS_SHELL_FILE}" ]; then . "${___MY_VMOPTIONS_SHELL_FILE}"; fi

# Kiro CLI post block. Keep at the bottom of this file.
[[ -f "${HOME}/Library/Application Support/kiro-cli/shell/zshrc.post.zsh" ]] && builtin source "${HOME}/Library/Application Support/kiro-cli/shell/zshrc.post.zsh"
export PATH="/usr/local/opt/python@3.10/bin:$PATH"
alias python3="/usr/local/opt/python@3.10/bin/python3"
alias python3="/usr/bin/python3"
alias python3="/usr/local/Cellar/python@3.10/"
alias python3="/usr/local/Cellar/python@3.10/3.10.20/bin/python3.10"
export PATH=/usr/local/opt/python@3.10/bin:$PATH
export PATH=/usr/local/opt/python@3.10/bin:$PATH
alias pip3="python3 -m pip"
alias python="/usr/local/opt/python@3.10/bin/python3"
alias python3="/usr/local/opt/python@3.10/bin/python3"
