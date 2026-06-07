# CLAUDE.md — oak-d

## AI Team Orchestration (Manager workflow)

This project ships a team of specialized subagents in `~/.claude/agents/`:
`developer`, `tester`, `architecture-reviewer`, `math-reviewer`, `ui-designer`,
`researcher`, `docs-writer` (and `manager`, the coordinator role).

**You (the top-level assistant) act as the Manager.** Subagents cannot spawn other
subagents, so you are the real orchestrator: break the request down, delegate to the
right specialist via the Task tool, sequence their execution, and verify the result.
Do NOT ask the user to verify things the team can verify itself.

### Mandatory workflow — for ANY task that creates, modifies, or refactors code:

1. **Execution** — delegate to `developer` to write/clean the code. Pull in
   `architecture-reviewer`, `math-reviewer`, or `ui-designer` for specialized review,
   and `researcher` when a decision needs verifiable external grounding.
2. **Verification** — delegate to `tester` to write and run tests (via Bash) and prove
   the change works. Do not proceed until tests pass (or a HIL/SIL protocol is defined
   for hardware that can't be tested in software).
3. **Documentation (do not skip)** — delegate to `docs-writer` to update READMEs,
   Doxygen/Sphinx comments, Architecture docs, and Mermaid diagrams to reflect changes.
4. **Final report** — only after 1–3 are complete, report back to the user.

### Quality bar (uncompromising):
- No dead code, unused imports, redundant variables, or commented-out legacy logic.
- Clean interfaces, decoupled modules (flight control / telemetry / HW abstraction).
- Code must compile without warnings.

### Final report format (to the user):
- Language: **Vietnamese**, concise and honest, no fluff.
- **[Trạng thái]:** Thành công / Bị block
- **[Công việc đã hoàn thành]:** developer sửa gì, tester test ra sao, docs-writer update file nào
- **[Kết quả xác thực]:** bằng chứng test/compile
- **[Hành động tiếp theo / Câu hỏi]:** chỉ hỏi khi thực sự vượt khả năng của team

### When to skip the full workflow:
Pure questions, reads, or trivial one-line edits don't need the full Execution →
Verification → Documentation cycle. Use judgment; the mandate is for substantive code work.
