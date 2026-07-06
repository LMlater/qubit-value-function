param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath,
    [Parameter(Mandatory=$true)]
    [string]$OutputPath
)

$word = $null
$doc = $null
try {
    $resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
    $outputDirectory = Split-Path -Parent $OutputPath
    if (!(Test-Path -LiteralPath $outputDirectory)) {
        New-Item -ItemType Directory -Path $outputDirectory | Out-Null
    }
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Open($resolvedInput, $false, $true)
    $doc.ExportAsFixedFormat($OutputPath, 17)
    Write-Output $OutputPath
}
finally {
    if ($doc -ne $null) {
        $doc.Close($false)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}
