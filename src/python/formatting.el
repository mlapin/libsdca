;; This buffer is for notes you don't want to save, and for Lisp evaluation.
;; If you want to create a file, visit that file with C-x C-f,
;; then enter the text in that file's own buffer.

;;; Code:
(defun custom-c++-mode-hook ()
  (setq c-basic-offset 2)
  (c-set-offset 'innamespace 0)
  (c-set-offset 'substatement-open 0)
  (c-set-offset 'template-args-cont 10)
  (c-set-offset 'arglist-intro 4)
  (c-set-offset 'arglist-close 4)
  (c-set-offset 'access-label 0)
  (c-set-offset 'topmost-intro 0)
  (c-set-offset 'topmost-intro-cont 0)
  (c-set-offset 'defun-block-intro 2)
  (c-set-offset 'statement-cont 0))
(add-hook 'c++-mode-hook 'custom-c++-mode-hook)

(defun custom-c-mode-hook ()
  (setq c-basic-offset 2)
  (c-set-offset 'innamespace 0)
  (c-set-offset 'substatement-open 0)
  (c-set-offset 'template-args-cont 10)
  (c-set-offset 'arglist-intro 4)
  (c-set-offset 'arglist-close 4)
  (c-set-offset 'access-label 0)
  (c-set-offset 'topmost-intro 0)
  (c-set-offset 'topmost-intro-cont 0)
  (c-set-offset 'defun-block-intro 2)
  (c-set-offset 'statement-cont 0))
(add-hook 'c-mode-hook 'custom-c-mode-hook)

;;; formatting.el ends here
