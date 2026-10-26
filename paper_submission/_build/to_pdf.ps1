# Exports a .docx to PDF with Microsoft Word (wdFormatPDF = 17).
# Usage: powershell -File to_pdf.ps1 <in.docx> <out.pdf>
param([string]$InFile, [string]$OutFile)
$in = (Resolve-Path $InFile).Path
$out = [System.IO.Path]::GetFullPath($OutFile)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($in, $false, $true)
    $doc.SaveAs2($out, 17)
    $doc.Close($false)
    Write-Output "saved $out"
} finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
