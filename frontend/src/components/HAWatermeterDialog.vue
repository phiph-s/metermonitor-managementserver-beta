<template>
  <n-modal v-model:show="showModal" preset="dialog" title="Add Watermeter from Home Assistant">
    <n-form ref="formRef" :model="formValue" :rules="rules">
      <n-form-item label="Watermeter Name" path="name">
        <n-input
          v-model:value="formValue.name"
          placeholder="e.g., living_room_meter"
          @keydown.enter.prevent
        />
      </n-form-item>

      <n-form-item label="Camera Entity" path="ha_entity_camera">
        <n-select
          v-model:value="formValue.ha_entity_camera"
          :options="cameraOptions"
          :loading="loadingEntities"
          placeholder="Select a camera entity"
          filterable
        />
      </n-form-item>

      <n-form-item label="LED/Light Entity (Optional)" path="ha_entity_led">
        <n-select
          v-model:value="formValue.ha_entity_led"
          :options="lightOptions"
          :loading="loadingEntities"
          placeholder="Select a light entity (optional)"
          filterable
          clearable
        />
      </n-form-item>

      <n-form-item label="Polling Frequency (seconds)" path="ha_frequency">
        <n-input-number
          v-model:value="formValue.ha_frequency"
          :min="10"
          :max="3600"
          placeholder="600"
        />
      </n-form-item>
    </n-form>

    <template #action>
      <n-space>
        <n-button @click="handleCancel">Cancel</n-button>
        <n-button type="primary" @click="handleSubmit" :loading="submitting">
          Create Watermeter
        </n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<script setup>
import { ref, watch, computed } from 'vue';
import { NModal, NForm, NFormItem, NInput, NSelect, NInputNumber, NButton, NSpace, useMessage } from 'naive-ui';

const props = defineProps({
  show: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(['update:show', 'created']);

const message = useMessage();
const host = import.meta.env.VITE_HOST;

const showModal = ref(props.show);
const loadingEntities = ref(false);
const submitting = ref(false);
const cameras = ref([]);
const lights = ref([]);

const formRef = ref(null);
const formValue = ref({
  name: '',
  ha_entity_camera: null,
  ha_entity_led: null,
  ha_frequency: 600
});

const rules = {
  name: {
    required: true,
    message: 'Please enter a name for the watermeter',
    trigger: 'blur'
  },
  ha_entity_camera: {
    required: true,
    message: 'Please select a camera entity',
    trigger: 'change'
  },
  ha_frequency: {
    required: true,
    type: 'number',
    message: 'Please enter a valid frequency',
    trigger: 'blur'
  }
};

// Watch for show prop changes
watch(() => props.show, (newVal) => {
  showModal.value = newVal;
  if (newVal) {
    loadEntities();
  }
});

// Watch for modal close
watch(showModal, (newVal) => {
  emit('update:show', newVal);
  if (!newVal) {
    resetForm();
  }
});

// Compute options for selects
const cameraOptions = computed(() => {
  return cameras.value.map(entity => ({
    label: entity.attributes?.friendly_name || entity.entity_id,
    value: entity.entity_id
  }));
});

const lightOptions = computed(() => {
  return lights.value.map(entity => ({
    label: entity.attributes?.friendly_name || entity.entity_id,
    value: entity.entity_id
  }));
});

// Load entities from HA
const loadEntities = async () => {
  loadingEntities.value = true;
  try {
    // Load cameras
    const cameraResponse = await fetch(host + 'api/ha/entities?entity_type=camera', {
      headers: {
        'secret': `${localStorage.getItem('secret')}`
      }
    });

    if (cameraResponse.ok) {
      const cameraData = await cameraResponse.json();
      cameras.value = cameraData.entities || [];
    }

    // Load lights
    const lightResponse = await fetch(host + 'api/ha/entities?entity_type=light', {
      headers: {
        'secret': `${localStorage.getItem('secret')}`
      }
    });

    if (lightResponse.ok) {
      const lightData = await lightResponse.json();
      lights.value = lightData.entities || [];
    }
  } catch (error) {
    message.error('Failed to load Home Assistant entities: ' + error.message);
  } finally {
    loadingEntities.value = false;
  }
};

// Handle form submission
const handleSubmit = async () => {
  try {
    await formRef.value?.validate();

    submitting.value = true;

    const response = await fetch(host + 'api/watermeters/ha', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'secret': `${localStorage.getItem('secret')}`
      },
      body: JSON.stringify({
        name: formValue.value.name,
        ha_entity_camera: formValue.value.ha_entity_camera,
        ha_entity_led: formValue.value.ha_entity_led || null,
        ha_frequency: formValue.value.ha_frequency
      })
    });

    if (response.ok) {
      const result = await response.json();
      message.success(`Watermeter "${result.name}" created successfully!`);
      showModal.value = false;
      emit('created', result);
    } else {
      const error = await response.json();
      message.error(error.detail || 'Failed to create watermeter');
    }
  } catch (error) {
    if (error?.errors) {
      // Validation error
      return;
    }
    message.error('Failed to create watermeter: ' + error.message);
  } finally {
    submitting.value = false;
  }
};

const handleCancel = () => {
  showModal.value = false;
};

const resetForm = () => {
  formValue.value = {
    name: '',
    ha_entity_camera: null,
    ha_entity_led: null,
    ha_frequency: 600
  };
  formRef.value?.restoreValidation();
};
</script>

<style scoped>
</style>

