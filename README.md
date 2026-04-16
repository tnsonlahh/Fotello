# Automated Image Enhancement System (Architecture/Real-Estate)

Pipeline nay duoc thiet ke de chay ngay, khong can training model.
Tap trung vao anh xay dung/nha o voi 3 buoc:

1. Undistortion + Geometry correction
2. Brightness adjustment (giam overexposure)
3. Color correction

## Tinh nang

- Tu dong can bang duong thang (auto straighten)
- Tu dong perspective correction (neu phat hien duoc tu giac lon)
- Giam do choi sang neu anh qua sang
- White balance theo gray-world
- Tang local contrast bang CLAHE (ruc ro)
- Tong hop saturation boost de mau sac ruc rong
- Cho phep chinh tay qua tham so CLI:
  - exposure
  - saturation
  - warmth

## Yeu cau

- Python 3.10+
- CPU la du; khong bat buoc GPU

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chay voi 1 anh

```powershell
python run_pipeline.py --input test01.jpeg --output .\output
```

## Chay batch ca thu muc

```powershell
python run_pipeline.py --input .\input --output .\output
```

## Vi du chinh tay

```powershell
python run_pipeline.py `
  --input .\input `
  --output .\output `
  --manual-exposure -0.1 `
  --manual-saturation 1.2 `
  --manual-warmth 0.1 `
  --clahe-clip 4.0
```

## Mo ta cac tham so quan trong

- `--no-straighten`: tat can bang duong thang
- `--no-perspective`: tat perspective correction
- `--no-auto-brightness`: tat giam overexposure
- `--target-luma`: muc sang muc tieu (mac dinh 0.52)
- `--no-auto-wb`: tat auto white balance
- `--no-clahe`: tat CLAHE
- `--clahe-clip`: CLAHE clip limit (mac dinh 3.5 - ruc ro)
- `--clahe-grid`: CLAHE grid size (mac dinh 8)
- `--saturation-boost`: saturation tu dong (mac dinh 1.35 - ruc ro)
- `--manual-exposure`: bu EV thu cong (-1.0..1.0)
- `--manual-saturation`: saturation thu cong (0.5..1.5)
- `--manual-warmth`: do am/mat mau thu cong (-1.0..1.0)

## Cau truc

- `run_pipeline.py`: entrypoint CLI
- `src/enhancer.py`: toan bo xu ly anh

## Luu y

- Perspective correction la phuong phap co dien (khong deep learning), nen voi mot so anh khong ro canh co the bo qua.
- Pipeline hien tai toi uu hoa: can thang + perspective correction + giam sang + white balance + CLAHE + saturation tong hop.
- Mau sac ruc ro nhat: clahe-clip 3.5 + saturation-boost 1.35 la muc tieu cho anh kien truc.
