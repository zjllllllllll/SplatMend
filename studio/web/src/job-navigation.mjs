export function rememberedJob(active, search, saved) {
    const requested = new URLSearchParams(search).get('job');
    const value = active || requested || saved;
    if (!value) return null;
    if (!/^[a-f0-9]{32}$/.test(value)) throw new Error('本地任务链接无效，请重新打开工作台。');
    return value;
}
