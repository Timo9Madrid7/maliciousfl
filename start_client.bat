@REM python clear_comb_client.py --id 1 | ^
@REM python clear_comb_client.py --id 0

for /l %%i in (1,1,9) do start /b python clear_comb_client.py --id %%i
python clear_comb_client.py --id 0
rundll32 user32.dll,MessageBeep