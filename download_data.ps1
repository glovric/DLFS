# Download data script for Windows

$DataDir = ".\data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

$Urls = @(
    "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset",
    "https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist",
    "https://www.kaggle.com/api/v1/datasets/download/awsaf49/clean-weather-dataset",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

foreach ($Url in $Urls) {
    $FileName = Split-Path $Url -Leaf

    if ($FileName -notmatch '\.(txt|zip)$') {
        $FileName += ".zip"
    }

    $OutputPath = Join-Path $DataDir $FileName

    Write-Host "Downloading $Url -> $OutputPath ..."
    curl.exe -L -o $OutputPath $Url
    Write-Host "✅ Done.`n"
}