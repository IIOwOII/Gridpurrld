# composition-gridworld
*Gridworld environment for performing composition tasks with Reinforcement Learning!*
<br/>

### Data Folder
환경 구성에 필요한 사진 파일들이 들어있습니다.<br/>
py파일이 존재하는 경로에 같이 존재해야합니다.<br/>

### GridWorld_Composition.py
실제 렌더링될 환경입니다.<br/>
방향키로 이동하며, 스페이스바로 아이템을 수집할 수 있습니다.<br/>
수집된 아이템은 인벤토리에 저장됩니다.<br/>
인벤토리에는 아이템이 최대 3개까지 저장되며, 인벤토리가 가득 차면 스테이지가 리셋됩니다.<br/>
(다음 스테이지로 넘어가는 것 구현 예정)<br/>

### GridWorld_Composition_Environment.py
RL 트레이닝용으로 변환한 환경입니다.<br/>
