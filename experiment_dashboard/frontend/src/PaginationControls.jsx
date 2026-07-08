import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

export function PaginationControls({ page, pages, onPageChange }) {
  const [draft, setDraft] = useState(String(page));
  const totalPages = Math.max(1, Number(pages) || 1);
  const currentPage = Math.min(Math.max(1, Number(page) || 1), totalPages);

  useEffect(() => {
    setDraft(String(currentPage));
  }, [currentPage]);

  const goToPage = (value) => {
    const next = Math.min(Math.max(1, Number.parseInt(value, 10) || 1), totalPages);
    onPageChange(next);
    setDraft(String(next));
  };

  const submit = (event) => {
    event.preventDefault();
    goToPage(draft);
  };

  return (
    <form className="pagination" onSubmit={submit}>
      <button
        className="icon-button"
        type="button"
        title="第一页"
        disabled={currentPage <= 1}
        onClick={() => goToPage(1)}
      >
        <ChevronsLeft size={18} />
      </button>
      <button
        className="icon-button"
        type="button"
        title="上一页"
        disabled={currentPage <= 1}
        onClick={() => goToPage(currentPage - 1)}
      >
        <ChevronLeft size={18} />
      </button>
      <span>第 {currentPage} / {totalPages} 页</span>
      <label className="page-jump">
        <span>跳转</span>
        <input
          type="number"
          min="1"
          max={totalPages}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => setDraft(String(currentPage))}
        />
      </label>
      <button className="button secondary page-jump-button" type="submit">
        跳转
      </button>
      <button
        className="icon-button"
        type="button"
        title="下一页"
        disabled={currentPage >= totalPages}
        onClick={() => goToPage(currentPage + 1)}
      >
        <ChevronRight size={18} />
      </button>
      <button
        className="icon-button"
        type="button"
        title="最后一页"
        disabled={currentPage >= totalPages}
        onClick={() => goToPage(totalPages)}
      >
        <ChevronsRight size={18} />
      </button>
    </form>
  );
}
