# Parallel Hill Climbing

PHCRegressor use graph representation of mathematical expressions
<table>
<tr>
<td> <b>Tree representation</b> </td> <td> <b>Graph representation</b> </td>
</tr>
<tr>
<td>

```mermaid
graph TD;
  mul-->sqrt;
  mul-->add;
  sqrt-->x1;
  add-->x2;
  add-->x3;
  ```

</td>
<td>

```mermaid
graph TD;
  mul-->sub;
  mul-->add;
  sub-->div;
  sub-->x3;
  add-->div;
  add-->x3;
  div-->x1;
  div-->x2;
  ```

</td>
</tr>
</table>

```cpp
struct Instruction
{
    uint32_t mSrc1;
    uint32_t mSrc2;
    uint32_t mOpCode;
};
template <typename T>
struct Code
{
    std::vector<T> mConstants;
    std::vector<Instruction> mCodeInstructions;
};
```
