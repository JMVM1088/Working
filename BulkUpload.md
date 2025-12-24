Step 1:

Get-ChildItem -Path 'C:\Users\jv2mk\OneDrive\Stock\HistoricalData_AI\ETF_1' -Filter 'BATS_*, 1D.csv' | ForEach-Object {
    $newName = $_.Name -replace '^BATS_(.*), 1D\.csv$', '$1.csv'
    Rename-Item -Path $_.FullName -NewName $newName 
}
