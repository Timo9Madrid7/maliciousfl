python clearAdaclipping_client_record.py --id 1 | ^
python clearAdaclipping_client_record.py --id 0

@REM for /l %%i in (1,1,9) do start /b python clearAdaclipping_client.py --id %%i
@REM python clearAdaclipping_client.py --id 0
@REM rundll32 user32.dll,MessageBeep