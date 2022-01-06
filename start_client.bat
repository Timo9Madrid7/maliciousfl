@REM python clear_comb_client_FedAvg.py --id 1 | ^
@REM python clear_comb_client_FedAvg.py --id 0

for /l %%i in (1,1,9) do start /b python clear_comb_client_FedAvg.py --id %%i
python inference_comb_client_FedAvg.py --id 0
rundll32 user32.dll,MessageBeep

@REM for /l %%i in (1,1,9) do start /b python clear_comb_client.py --id %%i
@REM python clear_comb_client.py --id 0
@REM rundll32 user32.dll,MessageBeep