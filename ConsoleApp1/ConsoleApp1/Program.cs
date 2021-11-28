// See https://aka.ms/new-console-template for more information
using System.Text;
using System.Text.RegularExpressions;
string? arr_str = null;
StringBuilder sb = new ();
while(!string.IsNullOrEmpty(arr_str = Console.ReadLine()))
{
    string res = Regex.Replace(arr_str, @"\s+", "\t");
    sb.Append(res);
}
Clipboard.GetClipboard().SetText(sb.ToString().Trim());