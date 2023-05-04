# to be sourced

shopt -s globstar

CFG_PATHS='configs/QED2.yaml'
OUT_PATHS='qed/'
UID_REGEX='[a-z0-9]{32}'
alias uidgrep='grep -oP "${UID_REGEX}"'


function orphans() {
    [ $? -eq 0 ] && comm -23 \
            <(find ${OUT_PATHS} | uidgrep | sort -u) \
            <(uidgrep -h ${CFG_PATHS} | sort -u)
    return $?
}

function assets(){
    while read uid; do
        find ${OUT_PATHS} -iname "${uid}*" 
    done
}

alias purge='assets | xargs rm -rfv'
alias quota='assets | xargs du -ch | tail -1 | cut -f 1'
