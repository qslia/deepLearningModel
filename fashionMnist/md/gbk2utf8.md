That error means your program is trying to print or write a Unicode character (`✓`, U+2713 check mark) but Windows by default is using the **GBK** code page, which cannot represent that character.

Here are some ways to fix it:

---

### 1. Change Python output encoding to UTF-8

At the very top of your script, force UTF-8 for stdout/stderr:

```python
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

---

### 2. Run Python with UTF-8 mode

Enable UTF-8 mode via environment variable or command-line flag:

* Command line:

  ```bat
  set PYTHONUTF8=1
  python yourscript.py
  ```
* Or:

  ```bat
  python -X utf8 yourscript.py
  ```

---

### 3. Change the console code page (cmd / PowerShell)

Before running:

```bat
chcp 65001
```

This switches the terminal to UTF-8 (`65001`). Then rerun your script.

---

### 4. Change log redirection encoding

If you only care about the log file (not terminal), open the log file with UTF-8 encoding in your Python code:

```python
with open("a.log", "w", encoding="utf-8") as f:
    f.write("✓ success\n")
```

---

✅ Recommendation:
If you’re redirecting with `>> a.log`, the simplest stable fix is to **force UTF-8 mode** (`set PYTHONUTF8=1`) and write logs with `encoding="utf-8"`.

---

Do you want the **output visible in the terminal** with checkmarks, or is it enough if the **log file** contains the correct Unicode characters?
