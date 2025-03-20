<template>

  <n-space vertical size="large">
    <n-layout>
      <n-layout-content content-style="padding: 24px;">
        <router-view></router-view>
      </n-layout-content>
    </n-layout>
  </n-space>

</template>

<script setup>
import {NLayout, NLayoutContent, NSpace, useNotification} from 'naive-ui';
import {onMounted, onUnmounted, ref} from "vue";

const alerts = ref([]);

const notification = useNotification();

const updateAlerts = async () => {
  alerts.value = await fetch(process.env.VUE_APP_HOST + 'api/alerts', {
    headers: {secret: localStorage.getItem('secret')}
  }).then(r => r.json());

  notification.destroyAll();
  for (const alert of Object.keys(alerts.value)) {
    notification.create({
      title: alert.toUpperCase(),
      content: alerts.value[alert],
      closable: false,
      type: 'error'
    });
  }
}
const interval = ref(null);
onMounted(() => {
  updateAlerts();
  interval.value = setInterval(updateAlerts, 60000);
});

onUnmounted(() => {
  clearInterval(interval.value);
});

</script>
<style>
body{
  background-color: #111111;
}

.apexcharts-tooltip {
  background: #f3f3f3;
  color: #292929;
}
</style>
