param()

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$envPath = Join-Path $repoRoot ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()

        if (-not $line -or $line.StartsWith("#")) { return }

        $parts = $line -split "=", 2
        if ($parts.Count -lt 2) { return }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim()

        if ($value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if (-not $key -or [string]::IsNullOrEmpty($value)) { return }

        # 이미 현재 세션에 값이 있으면 덮어쓰지 않음
        $existing = (Get-Item -Path ("env:" + $key) -ErrorAction SilentlyContinue).Value
        if ([string]::IsNullOrEmpty($existing)) {
            Set-Item -Path ("env:" + $key) -Value $value
        }
    }
}

if ([string]::IsNullOrEmpty($env:OPENAI_MODEL)) {
    $env:OPENAI_MODEL = "gpt-4o-mini"
}

streamlit run app/main.py
