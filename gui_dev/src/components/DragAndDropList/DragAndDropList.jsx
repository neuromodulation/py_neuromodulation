import { useRef, useState } from 'react'
import styles from './DragAndDropList.module.css'

export const DragAndDropList = () => {
  const predefinedOptions = [
    { id: 1, name: 'raw_resampling' },
    { id: 2, name: 'notch_filter' },
    { id: 3, name: 're_referencing' },
    { id: 4, name: 'preprocessing_filter' },
    { id: 5, name: 'raw_normalization' },
  ]

  const [options, setOptions] = useState([
    { id: 1, name: 'raw_resampling' },
    { id: 2, name: 'notch_filter' },
    { id: 3, name: 're_referencing' },
    { id: 4, name: 'preprocessing_filter' },
    { id: 5, name: 'raw_normalization' },
  ])

  const dragOption = useRef(0)
  const draggedOverOption = useRef(0)

  function handleSort() {
    const optionsClone = [...options]
    const temp = optionsClone[dragOption.current]
    optionsClone[dragOption.current] = optionsClone[draggedOverOption.current]
    optionsClone[draggedOverOption.current] = temp
    setOptions(optionsClone)
    console.log({ optionsClone })
  }

  function handleAdd(option) {
    setOptions([...options, option])
  }

  function handleRemove(id) {
    setOptions(options.filter(option => option.id !== id))
  }

  return (
    <main className={styles.dragDropList}>
      <h1 className={styles.title}>List</h1>
      {options.map((option, index) => (
        <div
          key={option.id}
          className={styles.item}
          draggable
          onDragStart={() => (dragOption.current = index)}
          onDragEnter={() => (draggedOverOption.current = index)}
          onDragEnd={handleSort}
          onDragOver={(e) => e.preventDefault()}
        >
          <p className={styles.itemText}>{option.name.replace("_", " ")}</p>
          <button className={styles.removeButton} onClick={() => handleRemove(option.id)}>Remove</button>
        </div>
      ))}
      <div className={styles.addSection}>
        <h2 className={styles.subtitle}>Add Elements</h2>
        {predefinedOptions.map(option => (
          <button
            key={option.id}
            className={styles.addButton}
            onClick={() => handleAdd(option)}
          >
            {option.name.replace("_", " ")}
          </button>
        ))}
      </div>
    </main>
  )
}