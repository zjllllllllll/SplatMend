[CmdletBinding()]
param([switch]$VerifyOnly)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$models = @(
    [pscustomobject]@{
        Name = 'SHARP'
        RelativePath = 'third_party\ml-sharp\ckpt\sharp_2572gikvuh.pt'
        Url = 'https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt'
        Size = [long]2809738232
        Sha256 = '94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d'
    },
    [pscustomobject]@{
        Name = 'LingBot-Depth v0.5'
        RelativePath = 'third_party\lingbot-depth\model\lingbot-depth\model.pt'
        Url = 'https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/resolve/79204ed6b837f4fdd192cf563e59481fecfa0295/model.pt?download=true'
        Size = [long]1284837952
        Sha256 = 'b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23'
    }
)

function Test-VerifiedFile([string]$Path, $Model) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $Model.Size) { return $false }
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $actual = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '')
        return [string]::Equals($actual, $Model.Sha256, [StringComparison]::OrdinalIgnoreCase)
    } finally {
        $stream.Dispose()
        $hasher.Dispose()
    }
}

function Invoke-ModelDownload([string]$CurlPath, [string]$Url, [string]$PartPath, [bool]$Resume) {
    $arguments = @(
        '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
        '--retry', '3', '--retry-delay', '3', '--connect-timeout', '30',
        '--speed-limit', '1024', '--speed-time', '120', '--output', $PartPath
    )
    if ($Resume) { $arguments += @('--continue-at', '-') }
    $arguments += $Url
    & $CurlPath @arguments
    return $LASTEXITCODE
}

try {
    $curl = (Get-Command curl.exe -ErrorAction Stop).Source
    foreach ($model in $models) {
        $destination = Join-Path $PSScriptRoot $model.RelativePath
        if (Test-VerifiedFile $destination $model) {
            Write-Host "$($model.Name): verified; download skipped."
            continue
        }
        if ($VerifyOnly) { throw "$($model.Name): missing or failed size/SHA256 verification: $destination" }

        $directory = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $partial = "$destination.download.tmp"
        if (Test-Path -LiteralPath $partial -PathType Leaf) {
            $partialSize = (Get-Item -LiteralPath $partial).Length
            if ($partialSize -gt $model.Size) {
                Remove-Item -LiteralPath $partial -Force
            } elseif ($partialSize -eq $model.Size -and -not (Test-VerifiedFile $partial $model)) {
                Remove-Item -LiteralPath $partial -Force
            }
        }

        if (-not (Test-VerifiedFile $partial $model)) {
            Write-Host "$($model.Name): downloading from the official model host..."
            $resume = Test-Path -LiteralPath $partial -PathType Leaf
            $downloadExit = Invoke-ModelDownload $curl $model.Url $partial $resume
            if ($downloadExit -eq 33 -and $resume) {
                Write-Host 'Server did not accept resume; restarting this file.'
                Remove-Item -LiteralPath $partial -Force
                $downloadExit = Invoke-ModelDownload $curl $model.Url $partial $false
            }
            if ($downloadExit -ne 0) {
                throw "$($model.Name): download failed (curl exit $downloadExit). Re-run to resume."
            }
            if (-not (Test-VerifiedFile $partial $model)) {
                Remove-Item -LiteralPath $partial -Force
                throw "$($model.Name): downloaded bytes failed size/SHA256 verification."
            }
        }

        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            [System.IO.File]::Replace($partial, $destination, $null)
        } else {
            [System.IO.File]::Move($partial, $destination)
        }
        Write-Host "$($model.Name): verified and installed."
    }
    Write-Host 'MODEL_WEIGHTS_READY'
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}