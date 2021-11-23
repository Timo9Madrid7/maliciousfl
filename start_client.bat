@REM python clearDitto_client.py --id 9 | ^
@REM python clearDitto_client.py --id 8 | ^
@REM python clearDitto_client.py --id 7 | ^
@REM python clearDitto_client.py --id 6 | ^
@REM python clearDitto_client.py --id 5 | ^
@REM python clearDitto_client.py --id 4 | ^
@REM python clearDitto_client.py --id 3 | ^
@REM python clearDitto_client.py --id 2 | ^
@REM python clearDitto_client.py --id 1 | ^
@REM python clearDitto_client.py --id 0

@REM python clear_dense_client.py --id 9 | ^
@REM python clear_dense_client.py --id 8 | ^
@REM python clear_dense_client.py --id 7 | ^
@REM python clear_dense_client.py --id 6 | ^
@REM python clear_dense_client.py --id 5 | ^
@REM python clear_dense_client.py --id 4 | ^
@REM python clear_dense_client.py --id 3 | ^
@REM python clear_dense_client.py --id 2 | ^
@REM python clear_dense_client.py --id 1 | ^
@REM python clear_dense_client.py --id 0

@REM python clearflguard_client.py --id 3 | ^
@REM python clearflguard_client.py --id 2 | ^
@REM python clearflguard_client.py --id 1 | ^
@REM python clearflguard_client.py --id 0

@REM python clearAdaclipping_client.py --id 9 | ^
@REM python clearAdaclipping_client.py --id 8 | ^
@REM python clearAdaclipping_client.py --id 7 | ^
@REM python clearAdaclipping_client.py --id 6 | ^
@REM python clearAdaclipping_client.py --id 5 | ^
@REM python clearAdaclipping_client.py --id 4 | ^
@REM python clearAdaclipping_client.py --id 3 | ^
@REM python clearAdaclipping_client.py --id 2 | ^
@REM python clearAdaclipping_client.py --id 1 | ^
@REM python clearAdaclipping_client.py --id 0

@REM python clear_comb_client.py --id 9 | ^
@REM python clear_comb_client.py --id 8 | ^
@REM python clear_comb_client.py --id 7 | ^
@REM python clear_comb_client.py --id 6 | ^
@REM python clear_comb_client.py --id 5 | ^
@REM python clear_comb_client.py --id 4 | ^
@REM python clear_comb_client.py --id 3 | ^
@REM python clear_comb_client.py --id 2 | ^
@REM python clear_comb_client.py --id 1 | ^
@REM python clear_comb_client.py --id 0

for /l %%i in (1,1,9) do start /b python clear_comb_client.py --id %%i
python clear_comb_client.py --id 0
rundll32 user32.dll,MessageBeep