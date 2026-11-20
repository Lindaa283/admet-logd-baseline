<#
Reproduce every number in this repository from a clean checkout.

Usage:
    .\run.ps1 build      # build the container image
    .\run.ps1 test       # run the test suite
    .\run.ps1 data       # download + curate the dataset
    .\run.ps1 train      # fit all models on both splits
    .\run.ps1 evaluate   # FINAL test evaluation (logged, run once)
    .\run.ps1 all        # build -> test -> data -> train

`evaluate` is intentionally excluded from `all`: touching the held-out test
set should be a deliberate act, and every run is recorded in
results/test_access_log.jsonl.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('build', 'test', 'data', 'train', 'evaluate', 'all')]
    [string]$Target
)

$ErrorActionPreference = 'Stop'
$Image = 'logd-baseline:0.1'
$Root = $PSScriptRoot

function Invoke-InContainer {
    param([string]$Script)
    docker run --rm -v "${Root}:/work" $Image sh -c $Script
    if ($LASTEXITCODE -ne 0) { throw "container command failed: $Script" }
}

switch ($Target) {
    'build' {
        docker build -t $Image $Root
        if ($LASTEXITCODE -ne 0) { throw 'image build failed' }
    }
    'test' { Invoke-InContainer 'python -m pytest' }
    'data' { Invoke-InContainer 'python -m logd.data --build' }
    'train' {
        Invoke-InContainer 'python -m logd.train --split scaffold'
        Invoke-InContainer 'python -m logd.train --split random'
    }
    'evaluate' {
        Invoke-InContainer 'python -m logd.evaluate --split scaffold --confirm-final-evaluation'
        Invoke-InContainer 'python -m logd.evaluate --split random --confirm-final-evaluation'
    }
    'all' {
        docker build -t $Image $Root
        if ($LASTEXITCODE -ne 0) { throw 'image build failed' }
        Invoke-InContainer 'python -m pytest'
        Invoke-InContainer 'python -m logd.data --build'
        Invoke-InContainer 'python -m logd.train --split scaffold'
        Invoke-InContainer 'python -m logd.train --split random'
        Write-Host ''
        Write-Host 'Training done. Run `.\run.ps1 evaluate` when development is finished.'
    }
}
