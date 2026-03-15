$py = "C:\Users\tejas\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
$dir = "C:\Users\tejas\OneDrive\Desktop\connectome_bpu"
$script = Join-Path $dir "run_experiments.py"

$tasks = @("MNIST", "FashionMNIST", "CIFAR10", "CartPole", "Audio", "SequentialMNIST")
foreach ($t in $tasks) {
    Start-Process -FilePath $py -ArgumentList "$script","--task","$t" -WorkingDirectory $dir -WindowStyle Hidden
    Write-Output "Launched runner for $t"
    Start-Sleep -Seconds 2
}
Write-Output "All 4 runners launched."
