// Restored tasks no longer have their original edit history. Preserve the whole
// saved deletion as one honest, reversible edit on the fresh working copy.
export function restoredDeletionOp(splat, ranges, deletedFlag) {
    return {
        name: 'restoreTaskDeletion', splat,
        async do() {
            this.splat.state.setBits(ranges, deletedFlag);
            await this.splat.updateState(deletedFlag);
        },
        async undo() {
            this.splat.state.clearBits(ranges, deletedFlag);
            await this.splat.updateState(deletedFlag);
        },
        destroy() { this.splat = null; }
    };
}
